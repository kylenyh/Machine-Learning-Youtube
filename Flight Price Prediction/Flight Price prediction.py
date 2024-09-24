import pandas as pd

main_dataset = pd.read_csv("C:/Users/user/OneDrive/Desktop/Flight Price Prediction Python/Clean_Dataset.csv") # file path directory of main dataset

main_dataset # shows main dataset

main_dataset.airline.value_counts() # counts frequency of each airline 

main_dataset.source_city.value_counts() # counts frequency of each source city

main_dataset.destination_city.value_counts() # counts frequency of each destination city

main_dataset.departure_time.value_counts() # counts frequency of each departure time

main_dataset.arrival_time.value_counts() # counts frequency of each arrival time

main_dataset.stops.value_counts() # counts frequency of each stop

main_dataset["class"].value_counts() # counts frequency of each class

main_dataset["duration"].min() # returns shortest flight time 

main_dataset["duration"].max() # returns longest flight time 

main_dataset["duration"].median() # returns median flight time 

## Preprocessing

main_dataset = main_dataset.drop("Unnamed: 0", axis = 1) # drops unnamed 0 column
main_dataset = main_dataset.drop('flight', axis = 1) # drops flight column

main_dataset["class"] =  main_dataset["class"].apply(lambda x: 1 if x == 'Business' else 0) # lambda binary encoding process for 0 and 1 return 

main_dataset.stops = pd.factorize(main_dataset.stops)[0] # adds factorized stops to dataset

main_dataset # prints dataset with factorised stops

main_dataset = main_dataset.join(pd.get_dummies(main_dataset.airline, prefix = "airline")).drop("airline", axis = 1) # adds adds one-hot encoded columns for the airline column to the main dataset, drops original airline column
main_dataset = main_dataset.join(pd.get_dummies(main_dataset.source_city, prefix = "source")).drop("source_city", axis = 1) # adds adds one-hot encoded columns for the source_city column to the main dataset, drops original source city column
main_dataset = main_dataset.join(pd.get_dummies(main_dataset.destination_city, prefix = "dest")).drop("destination_city", axis = 1) # adds adds one-hot encoded columns for the destination_city column to the main dataset, drops original destination city column
main_dataset = main_dataset.join(pd.get_dummies(main_dataset.arrival_time, prefix = "arrival")).drop("arrival_time", axis = 1) # adds adds one-hot encoded columns for the arrival_time column to the main dataset, drops original arrival time column
main_dataset = main_dataset.join(pd.get_dummies(main_dataset.departure_time, prefix = "departure")).drop("departure_time", axis = 1) # adds adds one-hot encoded columns for the departure_time column to the main dataset, drops original departure time column

main_dataset # shows new version of main dataset 

## Training Regression Model

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

X, y = main_dataset.drop("price", axis = 1), main_dataset.price # dropping price from X and putting in y

X # shows new dataset of X

y # shows new dataset of y

X_train, X_test, y_train, y_test  = train_test_split(X, y, test_size=0.2) # splits training and testing data 

reg = RandomForestRegressor(n_jobs=-1)

reg.fit(X_train, y_train) # Uses random forest to provide a regression fit 

reg.score(X_test, y_test) # gives regression score of X test and y test

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, root_mean_squared_error

y_pred = reg.predict(X_test) # predicts regression on x test

print("R2", r2_score(y_test, y_pred)) # Calculates R2 score
print("MSE", mean_squared_error(y_test, y_pred)) # Calculates MSE score
print("MAE", mean_absolute_error(y_test, y_pred)) # Calculates MAE score
print("RMSE", root_mean_squared_error(y_test, y_pred)) # Calculates RMSE score

import matplotlib.pyplot as plt

# plots a scatter polt of actual and predicted price 
plt.scatter(y_test, y_pred)
plt.xlabel("Actual Flight Price")
plt.ylabel("Predicted Flight Price")
plt.title("Prediction vs Actual Price")
plt.figure(figsize=(10, 6))

main_dataset.price.describe() # describes price in dataset

# creates a dictionary of feature importances from a RandomForestRegressor model and then sorts this dictionary by importance in descending order
importances = dict(zip(reg.feature_names_in_, reg.feature_importances_))
sorted_importances = sorted(importances.items(), key=lambda x:x[1], reverse=True)

sorted_importances

main_dataset.days_left.describe() # describes days left 

plt.figure(figsize=(15, 6))
plt.bar([x[0] for x in sorted_importances[:10]], [x[1] for x in sorted_importances[:10]]) # plots a histogram 

from sklearn.model_selection import GridSearchCV

forest = RandomForestRegressor()
reg = RandomForestRegressor(n_jobs=-1)

# performs hyperparameter tuning for a RandomForestRegressor model
# finds the optimal set of hyperparameters (settings) for a machine learning model to improve its performance

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 10, 20, 30],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
}

grid_search = GridSearchCV(reg, param_grid, cv=5)
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint

# performs a randomized hyperparameter search to optimize a RandomForestRegressor model using RandomizedSearchCV
# evaluates performance through cross-validation and retrieves the best model based on negative mean squared error

param_dist = {
    "n_estimators": randint(100, 300),
    "max_depth": [None, 10, 20, 30, 40, 50],
    "min_samples_split": randint(2, 11),
    "min_samples_leaf": randint(1, 5),
}

reg = RandomForestRegressor(n_jobs=-1)

random_search = RandomizedSearchCV(estimator=reg, param_distributions=param_dist, n_iter=2, cv=3,
                                   scoring="neg_mean_squared_error", verbose=2, random_state=10, n_jobs=-1)

random_search.fit(X_train, y_train)

best_regressor = random_search.best_estimator_

best_regressor.score(X_test, y_test) # gives a score on X test and y test
