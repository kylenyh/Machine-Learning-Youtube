import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

house_price_data  = pd.read_csv("C:/Users/user/OneDrive/Desktop/House Price Prediction Python/housing.csv") # reads dataset

house_price_data = house_price_data.dropna()

house_price_data # prints dataset

house_price_data.info() # summarises dataset

house_price_data.dropna() # drops na values in dataset

house_price_data.info() # summarizes the dataset after dropping na values

from sklearn.model_selection import train_test_split

X = house_price_data.drop(['median_house_value'], axis = 1) # contains all predictor variables except median house value
y = house_price_data['median_house_value'] # contains target variable which is median house value

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2) # splits X and y into training and testing sets 

train_data = X_train.join(y_train) # joins predictor variable and target variable together 

train_data

train_data.hist(figsize = (15, 8)) # produces a histogram of the trained data 

# selects only the numeric columns for correlation
numeric_train_data = train_data.select_dtypes(include=[float, int])

# computes the correlation matrix
correlation_matrix = numeric_train_data.corr()

# display the correlation matrix
correlation_matrix

plt.figure(figsize = (15, 8))

sns.heatmap(numeric_train_data.corr(), annot = True, cmap = "YlGnBu") # creates a heatmap of correlation matrix

# reduce the skewness of data to make it more normally distributed 
train_data["total_rooms"] = np.log(train_data["total_rooms"] + 1) # ensures that the transformation can be applied to data that may contain zeros
train_data["total_bedrooms"] = np.log(train_data["total_bedrooms"] + 1) # ensures that the transformation can be applied to data that may contain zeros
train_data["population"] = np.log(train_data["population"] + 1) # ensures that the transformation can be applied to data that may contain zeros
train_data["households"] = np.log(train_data["households"] + 1) # ensures that the transformation can be applied to data that may contain zeros

train_data.hist(figsize = (15, 8)) # produces a histogram of trained data based on trained variables above

pd.get_dummies(train_data.ocean_proximity) # performs one-hot encoding to convert these categorical variables into a format that can be provided to machine learning algorithms

train_data.join(pd.get_dummies(train_data.ocean_proximity)) # trains data from above

train_data = train_data.join(pd.get_dummies(train_data.ocean_proximity)).drop(['ocean_proximity'], axis = 1) # joins trained data into train data table 

train_data # shows table of new train data

plt.figure(figsize = (15, 8))

sns.heatmap(train_data.corr(), annot = True, cmap = "YlGnBu") # creates a heatmap of correlation matrix

plt.figure(figsize=(15, 8))
sns.scatterplot(x="latitude", y="longitude", data=train_data, hue="median_house_value", palette="coolwarm") # creates a scatterpolot of correlation matrix

train_data['bedroom ratio'] = train_data['total_bedrooms'] / train_data['total_rooms'] # calculates bedroom ratio from trained data 
train_data['household_rooms'] = train_data['total_rooms'] / train_data['households']  # calculates number of household rooms from trained data 

plt.figure(figsize = (15, 8))

sns.heatmap(train_data.corr(), annot = True, cmap = "YlGnBu") # creates a heatmap of correlation matrix

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_train, y_train = train_data.drop(['median_house_value'], axis=1), train_data['median_house_value']
X_train_s = scaler.fit_transform(X_train)

reg = LinearRegression()

reg.fit(X_train_s, y_train) # provides a regression model for the x train transformed data and y train data 


test_data = X_test.join(y_test) # join y test to x test 

test_data["total_rooms"] = np.log(test_data["total_rooms"] + 1) # ensures that the transformation can be applied to data that may contain zeros
test_data["total_bedrooms"] = np.log(test_data["total_bedrooms"] + 1) # ensures that the transformation can be applied to data that may contain zeros
test_data["population"] = np.log(test_data["population"] + 1) # ensures that the transformation can be applied to data that may contain zeros
test_data["households"] = np.log(test_data["households"] + 1) # ensures that the transformation can be applied to data that may contain zeros

test_data = test_data.join(pd.get_dummies(test_data.ocean_proximity)).drop(['ocean_proximity'], axis = 1)

test_data['bedroom ratio'] = test_data['total_bedrooms'] / test_data['total_rooms'] # calculates bedroom ratio in test data
test_data['household_rooms'] = test_data['total_rooms'] / test_data['households'] # calculates number of household rooms in test data

test_data # shwos table of new test data 

X_test, y_test = test_data.drop(['median_house_value'], axis=1), test_data['median_house_value'] # separates the input features and the target variable from the test_data dataframe

X_test # shows table of new x test data

from sklearn.ensemble import HistGradientBoostingRegressor


# uses histogram-based techniques to approximate the distribution of continuous features, which speeds up training and reduces memory usage

reg = HistGradientBoostingRegressor()
reg.fit(X_test, y_test)

# provides a regression fit of data 

X_test_s = scaler.transform(X_test) # standardizes or normalizes the features in the x test dataset

reg.score(X_test_s, y_test) # provides correlation score of x test and y test 

from sklearn.ensemble import RandomForestRegressor

forest = RandomForestRegressor()

forest.fit(X_train, y_train) # feeds trained data to random forest algorithm

forest.score(X_test_s, y_test) # provides forest score of x test and y test 

from sklearn.model_selection import GridSearchCV

forest = RandomForestRegressor()

# performs hyperparameter tuning for a RandomForestRegressor model
# finds the optimal set of hyperparameters (settings) for a machine learning model to improve its performance

param_grid = {
    "n_estimators": [100, 200, 300],
    "min_samples_split": [2, 4],
    "max_depth": [None, 4, 8]
}

grid_search = GridSearchCV(forest, param_grid, cv=5,
                           scoring="neg_mean_squared_error",
                           return_train_score=True)

grid_search.fit(X_train_s, y_train)
