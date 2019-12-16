#!/usr/bin/env python
# coding: utf-8

# # Import Libraries


import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error


# # Import Data



# Import data from files
df_train = pd.read_csv('training_data.csv')
df_test = pd.read_csv('test_data.csv')


# # Analyze Data



# Take a look at the data
df_train.head()




# Look at the unknown values and if such exist, clean them
df_train.isna().sum()




# Look at the unknown values and if such exist, clean them
df_test.isna().sum()




# Look at the joint distribution of columns in order to decide which algorithm to use for regression
sns.pairplot(df_train[["IDENTITY", "REGION","DAY","MONTH", "YEAR", "TRX_TYPE", "TRX_COUNT"]], diag_kind="kde")




# Look the statistics of data
statistics_train = df_train.describe()
statistics_train.pop("TRX_COUNT")
statistics_train = statistics_train.T
statistics_train


# # Random Forest Regression



# Extract the labels from the train set
train_labels = df_train.pop("TRX_COUNT")




# Define a regressor, I've extracted the parameters from tuning process, which is below.
# Time ETA: ~ 22 seconds 
regressor = RandomForestRegressor(n_estimators=400, max_depth = 50, min_samples_split=2,min_samples_leaf=1, random_state=0)
# Train the regressor
regressor.fit(df_train,train_labels)

# Get predictions for train set
predictions = regressor.predict(df_train)




# Look at the predictions as a sanity check (consistency with train_labels)
print("Predictions:", predictions)




# Calculate root mean squared error
rmse = np.sqrt(mean_squared_error(predictions, train_labels))

# Calculate mean absolute error
mae = mean_absolute_error(train_labels, predictions)




# Print rmse and mae
print("Root Mean Squared Error: ", rmse)
print("Mean Absolute Error: ", mae)




# Predict test data
predictions_test = regressor.predict(df_test)

# Look at the predictions as a sanity check (consistency with train_labels)
print("Predictions:", predictions_test)

# Write data into a csv file
np.savetxt('my_predictions_test_bugra.csv', predictions_test, delimiter=',', fmt='%f')


# # Hyper Parameter Tuning
# 



# Conduct a grid search in reaonable intervals with respect to rmse
n_estimators = [300,400,500]
max_depths = [50,100,150]
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
grid_search = [(n_est,depth,min_split,min_leaf) for n_est in n_estimators for depth in max_depths 
              for min_split in min_samples_split for min_leaf in min_samples_leaf]
best_params = None
best_rmse = -1
best_mae = -1

print("Starting parameter tuning...")
for (n_est,depth,min_split,min_leaf) in grid_search:
    regressor = RandomForestRegressor(n_estimators=n_est, max_depth = depth,min_samples_split=min_split 
                                      ,min_samples_leaf=min_leaf, random_state=0)
    regressor.fit(df_train,train_labels)
    predictions = regressor.predict(df_train)
    rmse = np.sqrt(mean_squared_error(predictions, train_labels))
    mae = mean_absolute_error(train_labels, predictions)
    if rmse > best_rmse:
        best_rmse = rmse
        best_mae = mae
        best_params = (n_est,depth,min_split,min_leaf)
    print("Params: " , "n_est: ", n_est,", max_depth: ", depth,", min_samples_split: ", 
      min_split,", min_samples_leaf: ",min_leaf)
    print("RMSE:", rmse)
    print("MAE:", mae)
    print()
#print("Tuning Finished.")
#n_est,depth,min_split,min_leaf = best_params
#print("Best Params: " , "n_est: ", n_est,", max_depth: ", depth,", min_samples_split: ", 
#      min_split,", min_samples_leaf: ",min_leaf)
#print("Best RMSE:", rmse)
#print("MAE: ", )






