# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 18:15:48 2019

@author: Helena
"""

import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from sklearn import tree
from sklearn.ensemble import AdaBoostRegressor
import warnings
warnings.filterwarnings("ignore")

# Create a constant value to test the data and get the average result
TIMES = 100

# Open the file
file_location = 'Behavior of the urban traffic of the city of Sao Paulo in Brazil.csv'
df = pd.read_csv(file_location)

# Get the original data info
df.info()

# The name of some columns are difficult to use, change them  
df.rename(columns={"Hour (Coded)": "time", 
                   "Slowness in traffic (%)": "slowness"}, inplace=True)
#print('\n')
#print(df.columns)

# Get the relationship of each attributes
feature_cols = ['slowness']
X = df.drop(feature_cols, axis=1, inplace=False)
y = df.slowness

#print(np.median(y))
#print(np.mean(y))
sns.heatmap(X.corr())
sns.pairplot(df, x_vars=X.columns, y_vars='slowness', kind='reg', height=4, aspect=1);
print("\n")

# From above actions we know that our data now is good to use
X = df.drop(columns = "slowness")
X = X.values
y = y*10
y = y.astype(int)
test_num = 0 - (int)(len(X)*0.2) 
y_list = []
y_list.append(y.values)
binary_y = [0 if e < np.median(y) else 1 for e in y]
y_list.append(binary_y)

for i in range(2):
    if i < 1:
        print("\n\nWith real number of y: ")
    else:
        print("\n\nWith binary number of y: ")
    y = y_list[i]    
    # ALGO 1:
    naive_bayes_list = []
    naive_bayes_list.append("GaussianNB")
    naive_bayes_list.append("BernoulliNB")
    naive_bayes_list.append("MultinomialNB")
    naive_bayes_error_list = []
    for i in range(len(naive_bayes_list)):
        naive_bayes_error_list.append(0.0) 
    # ALGO 2:
    knn_error_list = []
    for i in range(5):
        knn_error_list.append(0.0)
    # ALGO 3:
    regression_error = 0.0
    # ALGO 4:
    svm_list = []
    svm_list.append("linear")
    svm_list.append("rbf")
    svm_error_list = []
    for i in range(len(svm_list)):
        svm_error_list.append(0.0)
    # ALGO 5:
    mlp_list = []
    mlp_list.append("oneHiddenLayer")
    mlp_list.append("twoHiddenLayers")   
    mlp_error_list = []
    for i in range(len(mlp_list)):
        mlp_error_list.append(0.0)
    # ALGO 6:
    decisionTree_error = 0.0
    # ALGO 7:
    adaboost_error = 0.0
    
    # Start predicting    
    for times in range(TIMES):
        indices = np.random.permutation(len(X))
        # Separate the data to training and test
        X_train = X[indices[:test_num]]
        np_y = np.asarray(y)
        y_train = np_y[indices[:test_num]]
        X_test = X[indices[test_num:]]
        y_test = np_y[indices[test_num:]]
        # ALGO 1: Naive_Bayes   
        for i in range(len(naive_bayes_list)):
            if i == 0:
                clf = GaussianNB()
            elif i == 1:
                clf = BernoulliNB()
            else:
                clf = MultinomialNB()
            clf.fit(X_train,y_train)
            pred = clf.predict(X_test)
            error = mean_squared_error(y_test, pred)/len(y_test)
            if i == 0:
                naive_bayes_error_list[i] += error
            elif i == 1:
                naive_bayes_error_list[i] += error
            else:
                naive_bayes_error_list[i] += error
        
        # ALGO 2: KNN            
        for i in range(5):
            num = i+1
            neigh = KNeighborsClassifier(n_neighbors= num)
            neigh.fit(X_train, y_train) 
            pred = neigh.predict(X_test)   
            error = mean_squared_error(y_test, pred)/len(y_test)
            knn_error_list[i] += error
            
        # ALGO 3: Regression:         
        regr = linear_model.LinearRegression()
        regr.fit(X_train, y_train)
        pred = regr.predict(X_test)
        regression_error += mean_squared_error(y_test, pred)/len(y_test)
               
        # ALGO 4: SVM
        for i in range(len(svm_list)):
            svc = svm.SVC(kernel = svm_list[i])
            svc.fit(X_train, y_train)
            pred = svc.predict(X_test)
            error = mean_squared_error(y_test, pred)/len(y_test)
            svm_error_list[i] += error
            
        # ALGO 5: Neural Networks
        for i in range(len(mlp_list)):
            if i < 1:
                mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(10,), random_state=1)
            else:
                mlp = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(17,10), random_state=1)
            mlp.fit(X_train, y_train)
            pred = mlp.predict(X_test)
            error = mean_squared_error(y_test, pred)/len(y_test)
            mlp_error_list[i] += error
            
        # ALGO 6: 
        dtree = tree.DecisionTreeRegressor(max_depth=10, random_state=1)
        dtree.fit(X_train, y_train)        
        pred = dtree.predict(X_test)
        decisionTree_error += mean_squared_error(y_test, pred)/len(y_test)
               
        # ALGO 7:
        booster = AdaBoostRegressor(n_estimators=50)
        booster.fit(X_train, y_train)        
        pred = booster.predict(X_test)
        adaboost_error += mean_squared_error(y_test, pred)/len(y_test)
        
    # Print result
    times_str = str(TIMES)
    print("Try the algorithm of Naive_Bayes runs in " + times_str + " times: ")    
    for i in range(len(naive_bayes_error_list)):
        name = naive_bayes_list[i]
        print("The average mean squared error of " + name + " is:", naive_bayes_error_list[i]/TIMES)
    print("\nTry the algorithm of KNN runs in " + times_str + " times: ")
    for i in range(len(knn_error_list)):
        name = str(i+1)
        print("The average mean squared error of KNN with K = " + name + " is:", knn_error_list[i]/TIMES)
    print("\nTry the algorithm of Linear Regression runs in " + times_str + " times: ")    
    print("The average mean squared error of Linear Regression is:", regression_error/TIMES)    
    print("\nTry the algorithm of SVM runs in " + times_str + " times: ")    
    for i in range(len(svm_error_list)):
        name = svm_list[i]
        print("The average mean squared error of SVM kernel = " + name + " is:", svm_error_list[i]/TIMES)    
    print("\nTry the algorithm of MLP runs in " + times_str + " times: ") 
    for i in range(len(mlp_error_list)):
        name = mlp_list[i]
        print("The average mean squared error of mlp with " + name + " is:", mlp_error_list[i]/TIMES)     
    print("\nTry the algorithm of DecisionTree runs in " + times_str + " times: ") 
    print("The average mean squared error of DecisionTree is:", decisionTree_error/TIMES)    
    print("\nTry the algorithm of AdaBoost runs in " + times_str + " times: ") 
    print("The average mean squared error of AdaBoost is:", adaboost_error/TIMES)    
