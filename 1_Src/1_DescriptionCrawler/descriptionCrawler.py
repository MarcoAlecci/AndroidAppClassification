
# ## 🔢 Description Crawler


# Imports
import numpy    as np
import pandas   as pd
import google_play_scraper 
import langdetect
import datetime
import requests
import time
import json
import os


# #### Initialization


print("⚡ Start - {} ⚡\n".format(datetime.datetime.now()))
startTime = datetime.datetime.now()


# #### 📥 1) Load Data 


DATA_PATH = "../../0_Data/3_MalCatSet.csv"

# Read the data
appsDF = pd.read_csv(DATA_PATH)

# TEST
appsDF = appsDF.head(50)

print("--- #️⃣ Apps: {} ".format(appsDF.shape[0]))


appsDF.head(5)


# GP


def getGoogleEnglishDescription(pkgName):
	try:
		# Use googlePlayScraper Library
		result = google_play_scraper.app(pkgName, lang='en', country='us')

		if result is not None and result['description'] is not None:
			try:
				# Use langdetect
				lang = langdetect.detect(result['description'])
				# If the description is in English, return it
				if lang == "en":
					description = result['description'].replace('\n', ' ').replace('\r', '')
					return pd.Series([description], index=['description'])
				else:
					return pd.Series([np.nan], index=['description'])
			except langdetect.LangDetectException:
				return pd.Series([np.nan], index=['description'])
		else:
			return pd.Series([np.nan], index=['description'])
	except Exception:
		return pd.Series([np.nan], index=['description'])


appsDF['gpDescription'] = appsDF['pkgName'].apply(getGoogleEnglishDescription)
appsDF.head(3)


# AZ

def getDescriptionFromMetaDataAndroZoo(pkgName, retries=5, delay=10):
    url = 'https://androzoo.uni.lu/api/get_gp_metadata/{}'.format(pkgName)
    params = {'apikey': os.getenv('ANDROZOO_API_KEY')}
    
    attempt = 0
    while attempt < retries:
        response = requests.get(url, params=params)
        
        # Return Description
        if response.status_code == 200:
            return response.json()[0]['descriptionHtml']
        # Retry 
        elif response.status_code in [502, 503, 400]:
            attempt += 1
            print(f"Attempt {attempt} failed with status code {response.status_code}. Retrying in {delay} seconds...")
            time.sleep(delay)
        else:
            print(f"Request failed with status code {response.status_code}. No retries left.")
            return None
    
    print("Max retries exceeded. Request failed.")
    return None

appsDF['azDescription'] = appsDF['pkgName'].apply(getDescriptionFromMetaDataAndroZoo)
appsDF.head(3)



# Create the 'description' column
appsDF['description'] = appsDF['gpDescription'].fillna(appsDF['azDescription'])
print("--- #️⃣ Apps: {} ".format(appsDF.shape[0]))

# Drop rows where both 'gpDescription' and 'azDescription' are NaN
appsDF = appsDF.dropna(subset=['description'])
print("--- #️⃣ Apps: {} ".format(appsDF.shape[0]))

appsDF = appsDF.drop(columns=['gpDescription', 'azDescription'])
appsDF.head(3)



appsDF.to_csv("../TmpData/3_MalCatSet_DescriptionWithAZ.csv")


# ##### 🔚 End


endTime = datetime.datetime.now()
print("\n🔚 --- End - {} --- 🔚".format(endTime))

# Assuming endTime and startTime are in seconds
totalTime = endTime - startTime
minutes = totalTime.total_seconds() // 60
seconds = totalTime.total_seconds() % 60
print("⏱️ --- Time: {:02d} minutes and {:02d} seconds --- ⏱️".format(int(minutes), int(seconds)))


