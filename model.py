
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression

data = pd.read_csv("Test Dataset.csv")

# - Define the features and the target values
x = data.iloc[:, :-1].values
y = data.iloc[:, -1].values

# - We need to encode the categorical data (State [index 3] in this case)
ct = ColumnTransformer(transformers=[("encoder", OneHotEncoder(), [3])], remainder="passthrough")
x = np.array(ct.fit_transform(x))

# - Split the dataset to train and test data with a test size of 20% of the initial dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# - Fit and train the model
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# - Predict the test dataset and show the prediction next to the test poredictions
y_pred = regressor.predict(x_test)


# np.set_printoptions(precision=2) #show values with 2 decimal precision
# print(np.concatenate((y_pred.reshape(len(y_pred), 1), y_test.reshape(len(y_test), 1)), axis=1))




