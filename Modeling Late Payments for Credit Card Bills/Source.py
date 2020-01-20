#!/usr/bin/env python
# coding: utf-8



# Import Libraries
import pandas as pd
import numpy as np
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import Imputer
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
import csv
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from matplotlib import pyplot
from sklearn.svm import SVC
from sklearn import metrics



for index in range(1, 4):
    # Import Dataset
    X_train = pd.read_csv('hw07_target' + str(index) + '_training_data.csv')
    y_train =  pd.read_csv('hw07_target' + str(index) + '_training_label.csv')       
    X_test = pd.read_csv('hw07_target' + str(index) + '_test_data.csv')
    
    # Drop ID columns
    X_train.drop(["ID"],axis=1, inplace=True)
    X_test.drop(["ID"],axis=1, inplace=True)
    
    # Handle categorical data
    X_train['label'] = 'train'
    X_test['label'] = 'score'
    concat_df = pd.concat([X_train , X_test])
    features_df = None
    if index == 1:
        features_df = pd.get_dummies(concat_df,columns=['VAR45', 'VAR47', 'VAR75'])
    elif index == 2:
        features_df = pd.get_dummies(concat_df,columns=['VAR32', 'VAR65', 'VAR195'])
    else:
        features_df = pd.get_dummies(concat_df,columns=['VAR36', 'VAR153'])
  
    X_train = features_df[features_df['label'] == 'train']
    X_test = features_df[features_df['label'] == 'score']
    X_train = X_train.drop('label', axis=1)
    X_test = X_test.drop('label', axis=1)
    
    # Handle missing data
    fill_NaN = Imputer(missing_values=np.nan, strategy='mean', axis=1)
    imputed_DF = pd.DataFrame(fill_NaN.fit_transform(X_train))
    imputed_DF.columns = X_train.columns
    imputed_DF.index = X_train.index
    X_train = imputed_DF

    imputed_DF = pd.DataFrame(fill_NaN.fit_transform(X_test))
    imputed_DF.columns = X_test.columns
    imputed_DF.index = X_test.index
    X_test = imputed_DF
    
    # Standardized the data
    sc_X = StandardScaler()
    X_train_scaled = sc_X.fit_transform(X_train)
    X_train = pd.DataFrame(X_train_scaled)

    sc_X = StandardScaler()
    X_test_scaled = sc_X.fit_transform(X_test)
    X_test = pd.DataFrame(X_test_scaled)
    
    # Split the data for validation
    X_train_2, X_val, y_train_2, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=0)
    
    # Make predictions for validation set with AdaBoost Classifier
    abc = AdaBoostClassifier(n_estimators=200,
                         learning_rate=1)
    model = abc.fit(X_train_2, y_train_2["TARGET"])
    y_pred = model.predict_proba(X_val)
    posteriors_positive = y_pred[:, 1]
    
    # Plot the ROCAUC curve and save it as png file
    lr_auc = roc_auc_score(y_val["TARGET"], posteriors_positive)
    print("ROC AUC Score for Target #" + str(index))
    print('Logistic: ROC AUC=%.3f' % (lr_auc))
    lr_fpr, lr_tpr, _ = roc_curve(y_val["TARGET"], posteriors_positive)
    pyplot.plot(lr_fpr, lr_tpr, marker='.', label='Logistic')
    pyplot.xlabel('False Positive Rate')
    pyplot.ylabel('True Positive Rate')
    pyplot.legend()
    #pyplot.savefig("hw07_target" + str(index) + "_roc_auc_curve" + ".png")
    pyplot.show()
    
    # Make predictions for test set using full training set
    model = abc.fit(X_train, y_train["TARGET"])
    y_pred = model.predict_proba(X_test)
    posteriors_positive = y_pred[:, 1]
    
    # Save the posterior probabilities of belonging to positive(1) class
    with open('hw07_target' + str(index) + '_test_predictions.csv', mode='w') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['ID', 'TARGET'])
        start_index = 0
        if index == 1:
            start_index = 11000
        elif index == 2:
            start_index = 9000
        else:
            start_index = 5000
        for i in range(0, len(posteriors_positive)):
            writer.writerow([start_index + i, posteriors_positive[i]])






