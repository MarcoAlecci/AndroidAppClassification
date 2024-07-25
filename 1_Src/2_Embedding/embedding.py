# %% [markdown]
# ## 🔢 Description Embedding

# %%
# Imports
from   dotenv   import load_dotenv
import pandas   as pd
import datetime
import sys
import os

# Add the upper folder to sys.path
import LLMUtils
import Preprocessing

# %% [markdown]
# #### Initialization

# %%
print("⚡ Start - {} ⚡\n".format(datetime.datetime.now()))
startTime = datetime.datetime.now()

# %% [markdown]
# #### 📥 1) Load Data 

# %%
#DATA_PATH = "../../0_Data/0_AndroCatSet.csv"
DATA_PATH = "../../0_Data/5_NewMalCatSet.csv"

# Read the data
appsDF = pd.read_csv(DATA_PATH)
#appsDF = appsDF.drop(columns=['googlePlayCategoryID'])

# TEST
#appsDF = appsDF.head(2)

print("--- #️⃣ Apps: {} ".format(appsDF.shape[0]))

# %%
#appsDF = appsDF.head(5)

# %% [markdown]
# #### 🔢 2) Preprocess Description

# %%
appsDF['description'] = appsDF['description'].apply(Preprocessing.preprocessDescription)
appsDF.head(5)

# %% [markdown]
# #### 🔢 3) Generate Numerical Embeddings

# %%
# To interact with ChatGPT
gptManager = LLMUtils.GptManager()

# %% [markdown]
# $ Pay here

# %%
# Generate Embeddings
appsDF['embedding'] = appsDF['description'].apply(gptManager.generateEmbedding)
appsDF.head(5)

# %% [markdown]
# ### 🔢 4) Save into CSV

# %%
# MalCatSet
appsDF.to_csv("../TmpData/5_NewMalCatSetEmbeddings.csv", index=False)

# %% [markdown]
# ##### 🔚 End

# %%
endTime = datetime.datetime.now()
print("\n🔚 --- End - {} --- 🔚".format(endTime))

# Assuming endTime and startTime are in seconds
totalTime = endTime - startTime
minutes = totalTime.total_seconds() // 60
seconds = totalTime.total_seconds() % 60
print("⏱️ --- Time: {:02d} minutes and {:02d} seconds --- ⏱️".format(int(minutes), int(seconds)))


