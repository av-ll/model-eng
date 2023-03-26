# import necessary libraries including class Model from model.py file

import model
import pandas as pd
import warnings


warnings.simplefilter('ignore')

# read the data into a pandas DataFrame

data = pd.read_csv('prepared_data.csv')

# Split the data into features and target variable

X = data.iloc[:,:-1]

y = data.iloc[:,-1]

# we test some models after spliting the data into features and a target feature
# the method random_search of the Model class applies both Stratified KFold 
# (as we are dealing with a somewhat imbalanced dataset) to the data and Randomized Search
# to the model in order to find the best hyperparameters
# finally the method fit_model_StratKFold cross validates the model in order to
# get the average test f1 and test accuracy of the model

# Instantiate a logistic regression model with hyperparameters to test
from sklearn.linear_model import LogisticRegression

lr_par = {'penalty':('l1','l2','elasticnet',None),
          'C':(0.0001,0.1,0.3,0.6,0.9),
          'fit_intercept':(True,False),
          'solver':('lbfgs', 'newton-cg', 'newton-cholesky'),
          'l1_ratio':(0,0.2,0.5,0.8,1)}

lo_rg = model.Model('Logistic Regression',LogisticRegression(n_jobs=-1,random_state=0,max_iter=400),lr_par)

# Apply randomized search and cross-validation to the model

lo_rg.random_search(X, y)

lo_rg.cv_model_data(X, y)

# Check if Scaling improves the model's performance

lo_rg.random_search(X, y,use_scaling=True)

lo_rg.cv_model_scaled_data(X, y)


# Instantiate a Naive Bayes model
from sklearn.naive_bayes import GaussianNB

naive_bayes_params = {
    'var_smoothing': [1e-10,1e-09, 1e-08, 1e-07, 1e-06, 1e-05]
    }

naive_bayes = model.Model('Naive Bayes',GaussianNB(priors=None),naive_bayes_params)

# Apply randomized search and cross-validation to the model

naive_bayes.random_search(X, y)

naive_bayes.cv_model_data(X, y)

# Check if Scaling improves the model's performance

naive_bayes.random_search(X, y,use_scaling=True)

naive_bayes.cv_model_scaled_data(X, y)

# Instantiate a support vector machine model with hyperparameters to test

from sklearn.svm import LinearSVC

svm_params = {'penalty':['l1','l2'],
    'C': [0.01,0.1, 1, 10, 100],
    'loss': ['hinge', 'squared_hinge'],
    'tol':[1e-4,1e-3,1e-5,1e-7]
}

svm = model.Model('Support Vector Machine',LinearSVC(dual = False,random_state=0,max_iter=1500),svm_params)

# Apply randomized search and cross-validation to the model

svm.random_search(X, y)

svm.cv_model_data(X, y)

# Check if Scaling improves the model's performance

svm.random_search(X, y,use_scaling=True)

svm.cv_model_scaled_data(X, y)

from sklearn.tree import DecisionTreeClassifier

dt_param = {
    'criterion': ['gini','entropy','logloss'],
    'max_depth': [3, 5, 7, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 5],
    'max_features': [None, 'sqrt', 'log2']
}

dt = model.Model('Decision Tree', DecisionTreeClassifier(), dt_param)

# Apply randomized search and cross-validation to the model
dt.random_search(X, y)

dt.cv_model_data(X, y)

# Check if Scaling improves the model's performance

dt.random_search(X, y,use_scaling=True)

dt.cv_model_scaled_data(X, y)


# Instantiate a random forest model with hyperparameters to test

from sklearn.ensemble import RandomForestClassifier

# Create a parameter grid for Random Forest

rf_params = {
    'criterion': ['gini','entropy','logloss'],
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

rforest = model.Model('Random Forest', RandomForestClassifier(n_jobs=-1), rf_params) 

# Apply randomized search and cross-validation to the model

rforest.random_search(X, y)

rforest.cv_model_data(X, y)

# Check if Scaling improves the model's performance

rforest.random_search(X, y,use_scaling=True)

rforest.cv_model_scaled_data(X, y)

 
# Random Forest seems to overfit

# checking the results we can see that Logistic Regression and Support Vector Machines performed better than Naive Bayes

# and RandomForestClassifier which seem to be too complex of a model for the data

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,f1_score

# we fit each "winning" model with the best hyperparameters found in the previous steps and predict the labels of the test set
# we also calculate the accuracy and f1 score of each model on the test set

xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,random_state=27)

xtrain_scaled = scaler.fit_transform(xtrain)

xtest_scaled = scaler.transform(xtest)

lo_rg.best_model_scaled.fit(xtrain_scaled,ytrain)

lo_rg_pred = lo_rg.best_model_scaled.predict(xtest_scaled)

lo_rg_accuracy = accuracy_score(ytest,lo_rg_pred)

lo_rg_f1 = f1_score(ytest,lo_rg_pred)

svm.best_model_scaled.fit(xtrain_scaled,ytrain)

svm_pred = svm.best_model_scaled.predict(xtest_scaled)

svm_accuracy = accuracy_score(ytest,svm_pred)

svm_f1 =  f1_score(ytest,svm_pred)

print('lr_accuracy =',(lo_rg_accuracy*100).round(2),'%')
print('lr_f1 =',(lo_rg_f1).round(4))
print('svm_accuracy =',(svm_accuracy*100).round(2),'%')
print('svm_f1 =',(svm_f1).round(4))

print(lo_rg.best_model_scaled,svm.best_model_scaled,)

lo_rg.save_model('Logistic_Regression_Model.pkl',scaled_model=True)
svm.save_model('SVM_Model.pkl',scaled_model=True)









