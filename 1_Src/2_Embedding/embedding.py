# Imports
from   dotenv   import load_dotenv
import pandas   as pd
import itertools
import datetime
import argparse
import json
import sys
import os

# Add the upper folder to sys.path
sys.path.insert(0, "../../")
from   RedisClient import RedisClient
from   App         import App
from   App         import DataFlows
from   Embedding   import EmbeddingsManager

# TMP Folder
TMP_PATH = "../../../../0_Data/TMP/"

print("⚡ Start - {} ⚡\n".format(datetime.datetime.now()))
startTime = datetime.datetime.now()

# 
# Create TMP Folder
if not os.path.exists(TMP_PATH):
	os.makedirs(TMP_PATH)
	print("📁🆕 Folder created       :", TMP_PATH)
else:
	print("📁✅ Folder already exists:", TMP_PATH)


# Define the argument parser
parser = argparse.ArgumentParser(description='Script to configure embedding parameters.')
    # Add arguments
parser.add_argument('--dataset', type=str, choices=['0_AndroCatSet', '1_AndroCatSetMini', '2_AndroCatSetTestSet', '3_MalCatSet', '4_Mudflow'], required=True,
                    help='The dataset to process. Choose from available datasets.')
parser.add_argument('--embedding_model', type=str, choices=['gpt', 'codebert', 'sfr'], required=True,
                    help='The embedding model to use. Choose from "gpt", "codebert", or "sfr".')
parser.add_argument('--embedding_schema', type=str, choices=['onlysignatures', 'fullstatements'], required=True,
                    help='The embedding schema to use. Choose from "onlysignatures" or "fullstatements".')
# Parse the arguments
args = parser.parse_args()

# Assign the arguments to variables
EMBEDDING_MODEL  = args.embedding_model
EMBEDDING_SCHEMA = args.embedding_schema
DATASET          = args.dataset

# Print the values to verify they are being set correctly
print(f'--- ⚙️ DATASET: {DATASET}')
print(f'--- ⚙️ EMBEDDING_MODEL: {EMBEDDING_MODEL}')
print(f'--- ⚙️ EMBEDDING_SCHEMA: {EMBEDDING_SCHEMA}')

# Load .env file
load_dotenv()

# Determine Redis project key based on DATASET
if DATASET in ["0_AndroCatSet", "1_AndroCatSetMini", "2_AndroCatSetTestSet"]: 
    REDIS_PROJECT_KEY = "marco.dataflow.extraction.androcatset.backward.pairs"
	
elif DATASET == "3_MalCatSet": 
    REDIS_PROJECT_KEY = "marco.dataflow.extraction.malcatsetall.backward.pairs"
	
elif DATASET == "4_Mudflow": 
    REDIS_PROJECT_KEY = "marco.dataflow.extraction.mudflowall.backward.pairs"
	
else:
    raise ValueError("Unknown DATASET value provided")

print("\n--- 🗝️ REDIS KEY: {}".format(REDIS_PROJECT_KEY))

redisClientExtraction = RedisClient(host=os.getenv("REDIS_SERVER"), port=os.getenv("REDIS_PORT"), db=os.getenv("REDIS_DB"), password=os.getenv("REDIS_PSW"), projectKey = REDIS_PROJECT_KEY)


DATA_PATH = "../../../../0_Data/{}.csv".format(DATASET) 

# Read the data
appsDF = pd.read_csv(DATA_PATH)

# Print Number
print("--- #️⃣ Apps: {} ".format(appsDF.shape[0]))

# TEST
#appsDF = appsDF.head(10)

if EMBEDDING_SCHEMA == "onlysignatures":
	redisClientEmbedding  = RedisClient(host=os.getenv("REDIS_SERVER"), 
									port=os.getenv("REDIS_PORT"), 
									db=os.getenv("REDIS_DB"), 
									password=os.getenv("REDIS_PSW"), 
									projectKey = "marco.dataflow.embedding.onlysignatures")
			
if EMBEDDING_SCHEMA== "fullstatements":
	redisClientEmbedding  = RedisClient(host=os.getenv("REDIS_SERVER"), 
									port=os.getenv("REDIS_PORT"), 
									db=os.getenv("REDIS_DB"), 
									password=os.getenv("REDIS_PSW"), 
									projectKey = "marco.dataflow.embedding.fullstatements")

# 
# Create an Embedding Manager
embeddingsManager = EmbeddingsManager(redisClientEmbedding, EMBEDDING_MODEL)
print(embeddingsManager)

# 
def processRow(row):
		# Print message 
		print("\n--- 🔑 Analyzing APK: {} 🔑 ---".format(row['sha256']))

		# Create App instance
		app = App(row['sha256'], row['pkgName'], row['classID'])

		# Download Data Flows from Redis
		app.downloadDataFlowsFromRedis(redisClientExtraction)
		
		# Check if dataFlows have been extracted and are not empty.
		if(app.dataFlows is not None and not app.dataFlows.isEmpty()):
				
				# Keep only signature
				if EMBEDDING_SCHEMA == "onlysignatures":
					app.dataFlows.keepOnlySignatures()
					
				# Load DataFlows into Embeddings Manager
				embeddingsManager.loadDataFlowsFromApp(app.dataFlows)

# Apply the function to each row in the DataFrame
_ = appsDF.apply(processRow, axis=1)

# 
print(embeddingsManager)

embeddingsManager.generateMethodsEmbeddings(redisClientEmbedding, EMBEDDING_MODEL)

print(embeddingsManager)
if embeddingsManager.shape == 0:
	print("--- ⏭️ No NEW EMBEDDINGS Generated")

 
# ##### 🔚 End

# 
endTime = datetime.datetime.now()
print("\n🔚 --- End - {} --- 🔚".format(endTime))

# Assuming endTime and startTime are in seconds
totalTime = endTime - startTime
minutes = totalTime.total_seconds() // 60
seconds = totalTime.total_seconds() % 60
print("⏱️ --- Time: {:02d} minutes and {:02d} seconds --- ⏱️".format(int(minutes), int(seconds)))


