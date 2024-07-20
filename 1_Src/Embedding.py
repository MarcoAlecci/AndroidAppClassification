from   transformers         import AutoTokenizer, AutoModel
from   collections          import Counter
from   dotenv               import load_dotenv
from   torch                import Tensor
import torch.nn.functional  as F
import numpy                as np
import subprocess
import tiktoken
import requests
import openai
import shutil
import torch
import redis   
import json
import ast
import re
import os

import Utils

# Class to manage all the Embeddings together
class EmbeddingsManager:

	# Keep track of the number of apps Loaded
	numApps = None

	# Approach 1 --> First embed then combine.
	distinctMethods = None

	# Approach 2 --> First combine then embed the pairs as whole pairs.
	distinctPairs = None

	# Redis Client where to store Embedding
	redisClient = None

	# Manager and Shape
	embeddingModel = None
	manager 	   = None
	shape          = None


	# Initializer
	def __init__(self, redisClient, embeddingModel):
		self.distinctMethods       = set()
		self.distinctPairs         = set()
		self.redisClient           = redisClient
		self.numApps               = 0
		self.shape                 = 0

		# Select the embedding model
		self.embeddingModel = embeddingModel
		if embeddingModel == "gpt":
			self.manager = GptManager()
		elif embeddingModel == "codebert":
			self.manager = CodeBertManager()
		elif embeddingModel == "sfr":
			self.manager = SfrManager()
		else:
			print("\n--- ⚠️ Error: Unsupported embeddingModel type. Please use 'gpt', 'codebert', or 'sfr'.")


	def __str__(self):
		output = "\n--- ⭐ Embeddings Manager ⭐---\n"
		output += "--- #️⃣ Number of apps loaded       : {}\n".format(self.numApps)
		output += "--- #️⃣ Number of distinct methods  : {}\n".format(len(self.distinctMethods))
		output += "--- #️⃣ Number of distinct pairs    : {}\n".format(len(self.distinctPairs))

		output += "\n--- ☁️ REDIS Embedding Situation ☁️ ---\n"
		output += "--- ☁️ Embedding Model  : {}\n".format(self.embeddingModel)
		output += "--- ☁️ Number of method : {}\n".format(self.redisClient.getSize("{}.{}".format(self.redisClient.projectKey, self.embeddingModel)))
		output += "--- ☁️ Shape Embedding  : {}\n".format(self.shape)

		return output
	
	# Add new methods and pairs from an app dataFlows.
	def loadDataFlowsFromApp(self, dataFlows):
		# Add distinct Smali Methods
		self.distinctMethods.update(dataFlows.sources)
		self.distinctMethods.update(dataFlows.sinks)

		# Add Distinct Pairs
		for pair in dataFlows.pairs:
			jsonStr = json.dumps(pair, sort_keys=True)
			self.distinctPairs.add(jsonStr)      

		self.numApps += 1


	# Generate Embeddings and store them to REDIS.
	# Model --> "gpt", "codebert", "sfr" 
	def generateMethodsEmbeddings(self, redisClient, embeddingModel="gpt"):

		modelRedisKey = redisClient.projectKey + ".{}".format(embeddingModel)
		print("\n--- 🗝️ REDIS KEY: {}".format(modelRedisKey))

		for method in self.distinctMethods:
			print("\n---------------")
			print("--- 🌊 Method: {}".format(method))

			# Skip if already processed
			if redisClient.client.hget(modelRedisKey, method) is not None:
				print("--- ⏭️ Already Processed --> Skip")
				continue
			
			else:
				try:
					print("--- ▶️ Model: {}".format(embeddingModel))
					emb = self.manager.generateEmbedding(method)
					self.shape = len(emb)
						
					# To string
					embString = ','.join([str(f) for f in emb])

					# Push to Redis
					redisClient.client.hset(modelRedisKey, method, embString)

					# Print message SUCCESS
					print("--- ✅ Success for method: {}".format(method), flush=True)
					
				# Print message FAIL    
				except Exception as e:
					print("--- ❌ Failed with Exception {}".format(e), flush=True)

		print("-"*40)



class GptManager:

	# Open AI Client
	client = None

	model     = None
	price     = None
	tokenizer = None
		
	def __init__(self , model = "text-embedding-3-small", price = 0.02, tokenizer = "cl100k_base"):
		
		# Client Creation
		load_dotenv()
		apiKey = os.getenv("OPENAI_API_KEY")
		if apiKey is None:
			raise ValueError("--- ⚠️ OPENAI_API_KEY environment variable is not set")
		self.client    = openai.OpenAI(api_key = apiKey)

		# Details
		self.model     = model
		self.price     = price
		self.tokenizer = tokenizer

	# Generate Embeddings using OpenAi API        
	def generateEmbedding(self, inputData):
		# Remove new line chars
		inputData = inputData.replace("\n", " ")

		# Return Embedding
		return self.client.embeddings.create(input = [inputData], model = self.model).data[0].embedding
	
	# Count num of Tokens
	def getNumTokens(self, prompt):
		# "cl100k_base" --> the tokenizer used by GPT 3.5
		encoding = tiktoken.get_encoding(self.tokenizer)
		
		# Get the number of tokens
		numTokens = len(encoding.encode(prompt))
		return numTokens

	# Computer Cost Estimation
	def computeCost(self, data):
		# Get num of tokens
		totTokens = 0
		for string in data:
			numTokens =  self.getNumTokens(str(string))
			totTokens += numTokens

		# Compute cost
		totCost = totTokens / 10e6 * self.price

		return totTokens, totCost
	

class CodeBertManager:

	tokenizer = None
	model     = None

	# https://huggingface.co/microsoft/codebert-base
	def __init__(self):
		self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
		self.model     = AutoModel.from_pretrained("microsoft/codebert-base")


	# Return a list
	def generateEmbedding(self, inputData):
		# Tokenize input code
		inputs = self.tokenizer(inputData, return_tensors="pt", padding=True, truncation=True, max_length=1024)

		# Forward pass
		with torch.no_grad():
			outputs = self.model(**inputs)

		# Get the output embeddings
		last_hidden_state = outputs.last_hidden_state

		# Perform pooling to get fixed-length embeddings
		pooled_embeddings = torch.mean(last_hidden_state, dim=1)

		# Convert embeddings to numpy array
		embeddingsArray = pooled_embeddings.numpy()

		return embeddingsArray.tolist()[0]


class SfrManager:

	tokenizer = None
	model     = None

	# https://huggingface.co/Salesforce/SFR-Embedding-2_R
	def __init__(self):
		# Load tokenizer and model
		self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-2_R")
		self.model = AutoModel.from_pretrained("Salesforce/SFR-Embedding-2_R")

	def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
		left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
		if left_padding:
			return last_hidden_states[:, -1]
		else:
			sequence_lengths = attention_mask.sum(dim=1) - 1
			batch_size = last_hidden_states.shape[0]
			return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

	# Return List
	def generateEmbedding(self, inputData):
		# Tokenize the input text
		batch_dict = self.tokenizer(inputData, max_length=4096, padding=True, truncation=True, return_tensors="pt")
		
		# Get the model outputs
		outputs = self.model(**batch_dict)
		
		# Pool the embeddings from the last token
		embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
		
		# Convert embeddings to numpy array
		embeddingsArray = embeddings.detach().numpy()
		
		return embeddingsArray.tolist()[0]


# class SfrMistralManager:

#     tokenizer = None
#     model     = None

#     # https://huggingface.co/Salesforce/SFR-Embedding-Mistral
#     def __init__(self):
#         # Load tokenizer and model
#         self.tokenizer = AutoTokenizer.from_pretrained("Salesforce/SFR-Embedding-Mistral")
#         self.model     = AutoModel.from_pretrained("Salesforce/SFR-Embedding-Mistral")

#     def last_token_pool(self, last_hidden_states: Tensor, attention_mask: Tensor) -> Tensor:
#         left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
#         if left_padding:
#             return last_hidden_states[:, -1]
#         else:
#             sequence_lengths = attention_mask.sum(dim=1) - 1
#             batch_size = last_hidden_states.shape[0]
#             return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]

#     # Return List
#     def generateEmbedding(self, inputData):
#         # Tokenize the input text
#         batch_dict = self.tokenizer(inputData, max_length=4096, padding=True, truncation=True, return_tensors="pt")
		
#         # Get the model outputs
#         outputs = self.model(**batch_dict)
		
#         # Pool the embeddings from the last token
#         embeddings = self.last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])
		
#         # Convert embeddings to numpy array
#         embeddingsArray = embeddings.detach().numpy()
		
#         return embeddingsArray.tolist()[0]