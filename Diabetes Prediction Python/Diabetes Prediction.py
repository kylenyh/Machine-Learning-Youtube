import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

diabetes_data = pd.read_csv("C:/Users/user/OneDrive/Desktop/Diabetes Prediction Python/diabetes.csv") # file directory of data

diabetes_data # shows dataframe of data

diabetes_data.info() # prints info of data 

diabetes_data.describe() # describes info of data

diabetes_data.drop(["DiabetesPedigreeFunction", "Pregnancies"], axis=1, inplace=True) # drops columns from data 
 
diabetes_data # shows new dataframe of data

diabetes_data["Outcome"] = diabetes_data["Outcome"].astype("category", copy=False) # filters diagnosis for diabetic and non-diabetic females 
diabetes_data["Outcome"].value_counts().plot(kind="bar") # shows frequency counts 

plt.scatter(diabetes_data.BloodPressure, diabetes_data.Outcome) # logistic regression plot between blood pressure and diabetes as e.g.

# Add labels, title and legend
plt.xlabel('Blood Pressure')
plt.ylabel('Probability of Diabetes (0 = No Diabetes, 1 = Diabetes)')
plt.title('Logistic Regression Model for Diabetes Prediction')
plt.legend()

# divide into target variable and predictors

y = diabetes_data["Outcome"] # our target variable
X = diabetes_data.drop(["Outcome"], axis=1) # our predictor variable

X # shows data of X

y # shows data of y

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
