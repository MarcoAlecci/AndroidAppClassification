from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import datetime
import ast
import os
import sys

import warnings
warnings.filterwarnings("ignore")

# %% [markdown]
# #### Initialization

# %%
RANDOM_SEED = 151836

# %%
print("⚡ Start - {} ⚡\n".format(datetime.datetime.now()))
startTime = datetime.datetime.now()

# %% [markdown]
# #### 📥 1) Load Data 

# %%
DATA_PATH = "../TmpData/0_AndroCatEmbeddings.csv"

# Read the data
appsDF = pd.read_csv(DATA_PATH)

print("--- #️⃣ Apps: {} ".format(appsDF.shape[0]))

# %%
appsDF.head(5)

# %% [markdown]
# #### 2) Reorganize a bit the data

# %%
# Reorder columns
appsDF = appsDF[[col for col in appsDF.columns if col != "classID"] + ["classID"]]

# Rename the column
appsDF.rename(columns={"classID": "trueLabel"}, inplace=True)

# %%
def convertToNumpyArray(arrayStr):
	# Convert the string representation to a list using ast.literal_eval
	arrayList = ast.literal_eval(arrayStr)
	# Convert the list to a numpy array
	return np.array(arrayList)

# Apply the conversion function to the column
appsDF['embedding'] = appsDF['embedding'].apply(convertToNumpyArray)

# %%
appsDF.head(5)

# %% [markdown]
# #### 3) Train

# %% [markdown]
# Get X (Data), Y (Labels) And split.

# %%
# Convert the list of arrays into a NumPy matrix
X = np.vstack(appsDF['embedding'].tolist())
print("--- 📐 X Shape : {}".format(X.shape))

trueLabels = appsDF['trueLabel'].values 
print("--- 📐 Y Shape : {}".format(len(trueLabels)))

# Create an instance of LabelEncoder using a special value for NO CLASS
labelEncoder = LabelEncoder()
labelEncoder.fit(np.append(trueLabels, "NO_CLASS"))

# Get Encoded True Lalebsl
Y_True = labelEncoder.transform(trueLabels)
print(Y_True)

NO_CLASS_VALUE = labelEncoder.transform(["NO_CLASS"])[0]
print("No Class Value:", NO_CLASS_VALUE)

# Split the data into training and testing sets
X_train, X_test, Y_train, Y_test = train_test_split(X, Y_True, test_size=0.1, random_state = RANDOM_SEED)

# %%
# Define the parameter grid for grid search
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['linear', 'rbf', 'poly', 'sigmoid']
}

# Initialize the SVM model
modelName = "svm"
svm_model = SVC(probability=True, random_state=RANDOM_SEED)

# Initialize GridSearchCV
grid_search = GridSearchCV(svm_model, param_grid, refit=True, verbose=3, cv=5, n_jobs=-1)

# Fit GridSearchCV
grid_search.fit(X_train, Y_train)

# Get the best parameters and the best model
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

print("--- Best Parameters:\n", best_params)

# %%
# TO evaluate the model
def evaluateModel(modelName, Y_test, Y_pred, threshold):

    mask = (Y_pred != NO_CLASS_VALUE) & (Y_test != NO_CLASS_VALUE)
    Y_test_filtered = Y_test[mask]
    Y_pred_filtered = Y_pred[mask]

    # print(Y_test)
    # print(Y_test_filtered)
    # print(Y_pred)
    # print(Y_pred_filtered)

    # Compute metrics only on filtered labels
    accuracy = accuracy_score(Y_test_filtered, Y_pred_filtered)
    precision = precision_score(Y_test_filtered, Y_pred_filtered, average='weighted')
    recall = recall_score(Y_test_filtered, Y_pred_filtered, average='weighted')
    f1 = f1_score(Y_test_filtered, Y_pred_filtered, average='weighted')

    # Print metrics in a well-formatted way
    print("--- 📊 Model Evaluation Metrics 📊 ---")
    print("--- Model                    : {}".format(modelName))
    print("--- Confidence Threshold     : {:.2f}".format(threshold))
    print("--- Metrics:")
    print("------ Accuracy              : {:.4f}".format(accuracy))
    print("------ Precision (weighted)  : {:.4f}".format(precision))
    print("------ Recall (weighted)     : {:.4f}".format(recall))
    print("------ F1 Score (weighted)   : {:.4f}".format(f1))
    print("--"*20 + "\n")

# To apply a Threshold to the result confidence level
def applyThreshold(probabilities, threshold):
    Y_pred = []
    for prob in probabilities:
        if np.max(prob) < threshold:
            Y_pred.append(NO_CLASS_VALUE)  
        else:
            Y_pred.append(np.argmax(prob)) 
    return np.array(Y_pred)

# %%
# # Define a confidence threshold
# CONFIDENCE_THRESHOLD = 0.75

# # Get prediction probabilities
# probabilities = model.predict_proba(X_test)

# # Apply threshold to get final predictions
# Y_pred = applyThreshold(probabilities, CONFIDENCE_THRESHOLD)

# evaluateModel(modelName,Y_test, Y_pred, CONFIDENCE_THRESHOLD)

# predictedLabels = applyThreshold(model.predict_proba(X), CONFIDENCE_THRESHOLD)

# # Convert numeric predictions back to labels
# predictedLabels = labelEncoder.inverse_transform(predictedLabels)

# # Add the predictions to the DataFrame
# appsDF[modelName+ str(CONFIDENCE_THRESHOLD)] = predictedLabels

# %%
# Define a range of confidence thresholds
thresholds = [i/100 for i in range(50, 100, 5)]

# Get prediction probabilities
probabilities = best_model.predict_proba(X_test)

# Iterate through each threshold
for threshold in thresholds:
   
    # Apply threshold to get final predictions
    Y_pred = applyThreshold(probabilities, threshold)

    evaluateModel(modelName,Y_test, Y_pred, threshold)

    predictedLabels = applyThreshold(best_model.predict_proba(X), threshold)

    # Convert numeric predictions back to labels
    predictedLabels = labelEncoder.inverse_transform(predictedLabels)

    # Add the predictions to the DataFrame
    appsDF[modelName+ str(threshold)] = predictedLabels

# %%
# 'NN': MLPClassifier(solver='adam', alpha=1e-5, hidden_layer_sizes=(64,), random_state=RANDOM_SEED, max_iter=500)

# %% [markdown]
# ### 4) Save Results

# %%
appsDF.head(5)

# %%
appsDF = appsDF.drop(columns=['description','embedding'])
appsDF.to_csv("../TmpData/0_AndroCatSetNewLabels.csv", index=False)

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


