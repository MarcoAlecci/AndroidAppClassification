from   dotenv               import load_dotenv
import openai
import os

class GptManager:

	# Open AI Client Details
	client    = None
	model     = None
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
		self.tokenizer = tokenizer

	# Generate Embeddings using OpenAi API        
	def generateEmbedding(self, inputData):
		# Remove new line chars
		inputData = inputData.replace("\n", " ")

		# Return Embedding
		return self.client.embeddings.create(input = [inputData], model = self.model).data[0].embedding