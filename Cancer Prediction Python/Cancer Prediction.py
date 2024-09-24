import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

cancer_data = pd.read_csv('C:/Users/user/OneDrive/Desktop/Cancer Prediction Python/data.csv') # file directory of data

cancer_data # shows dataframe of data

cancer_data.info() # prints info of data 

cancer_data.describe() # describes info of data

sns.heatmap(cancer_data.isnull()) # creates heatmap of data

cancer_data.drop(["Unnamed: 32", "id"], axis=1, inplace=True) # drops columns from data 
 
cancer_data # shows new dataframe of data

cancer_data.diagnosis = [1 if value == "M" else 0 for value in cancer_data.diagnosis] # 1 for Males, 0 for Females 

cancer_data # shows new dataframe of data

cancer_data["diagnosis"] = cancer_data["diagnosis"].astype("category", copy=False) # filters diagnosis for males and females 
cancer_data["diagnosis"].value_counts().plot(kind="bar") # shows frequency counts 

# divide into target variable and predictors

y = cancer_data["diagnosis"] # our target variable
X = cancer_data.drop(["diagnosis"], axis=1)

X # shows data of X

from sklearn.preprocessing import StandardScaler

# create a scaler object
scaler = StandardScaler()

# fit the scaler to the data and transform the data 
X_scaled = scaler.fit_transform(X)

## Split the data

from sklearn.model_selection import train_test_split

# does train test split, 30% testing data, 70% training data 
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state = 42)

## Train the model

from sklearn.linear_model import LogisticRegression

# create the lr model
lr = LogisticRegression()

# train the model on the training data
lr.fit(X_train, y_train)

# predict the target variable on test data
y_pred = lr.predict(X_test)

## Evaluation of the model

from sklearn.metrics import accuracy_score

# prints accuracy score of data 
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy: .2f}")

from sklearn.metrics import classification_report

# prints classification report of data 
print(classification_report(y_test, y_pred))

