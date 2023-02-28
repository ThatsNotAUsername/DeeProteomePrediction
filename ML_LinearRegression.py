# -*- coding: utf-8 -*-
"""
Created on Tue Sep  7 17:04:38 2021

@author: roehla

Some ML functions where I return errors and predicted y

Not pretty, but helpful


"""


import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso, ElasticNet, Ridge
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest, f_regression#, mutual_info_regression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import sys
sys.path.append('../HelpFunctions')
from random import randrange
from sklearn.decomposition import PCA


# feature selection
def select_features(X_train, y_train, X_test, number_best_features):
	# configure to select all features
    
    fs = SelectKBest(score_func=f_regression, k=number_best_features)
    # fs = SelectKBest(score_func=mutual_info_regression, k=number_best_features)
	# learn relationship from training data
    fs.fit(X_train, y_train)
        
	# transform train input data
    X_train_fs = fs.transform(X_train)
	# transform test input data
    X_test_fs = fs.transform(X_test)
    return X_train_fs, X_test_fs, fs

def do_regression(X,y, number_best_features, n_splits, what_sort, alpha, l1_ratio=0.5, do_PCA=False, kernel='rbf'):
    
    print('alpha ', alpha)
    print(what_sort)
    random_state = randrange(100)
    all_rmse_test = []
    all_rmse_train = []
    all_r2_test = []
    all_r2_train = []
    all_y_predict_test = []
    all_y_predict_train = []
    all_y_test = []
    all_y_train = []
    all_best_features = {}
    all_fs ={}
    all_coefs = {}
    all_features = {}
    index_dictionaries = 0
    kf = KFold(n_splits, shuffle=True)
    for  train_index, test_index in kf.split(X):
        
        X_train = X.iloc[train_index]
        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform( X.iloc[test_index])
        if do_PCA: 
            n_components =min(20, len(X))
            pca = PCA(n_components=n_components).fit(X_train)
            X_train = pca.transform(X_train)
            X_test = pca.transform(X_test)        
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        y_train = y_train.apply(pd.to_numeric)
        
        if what_sort=='SimpleRegression':
            model = LinearRegression()
        elif what_sort=='Lasso':
            model = Lasso(alpha=alpha, random_state=random_state)
        elif what_sort=='ElasticNet':
            model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
        elif what_sort=='Ridge':
            model = Ridge(alpha=alpha, random_state=random_state)
        elif what_sort=='SVM':
            model = svm.SVR(kernel=kernel, C=alpha)
        reg = model.fit(X_train, y_train)
        # evaluate the model
        y_predict_test = model.predict(X_test)
        y_predict_train = model.predict(X_train)
            
        # evaluate predictions
        
        r2_test = reg.score(X_test,y_test)
        r2_train = reg.score(X_train,y_train)
        
        all_r2_test.append(r2_test)
        all_r2_train.append(r2_train)
        
        rmse_test = mean_squared_error(y_test, y_predict_test, squared=False)
        rmse_train = mean_squared_error(y_train, y_predict_train, squared=False)
        
        all_rmse_test.append(rmse_test)
        all_rmse_train.append(rmse_train)
        all_y_predict_test.append(y_predict_test)
        all_y_predict_train.append(y_predict_train)
        all_y_test.append(y_test)
        all_y_train.append(y_train)
    
        if what_sort=='SVM':
            all_coefs[index_dictionaries] = reg.support_
        else:
            all_coefs[index_dictionaries] = reg.coef_
        index_dictionaries += 1
    mean_rmse_test = np.mean(all_rmse_test)
    mean_r2_test = np.mean(all_r2_test)
    mean_rmse_train = np.mean(all_rmse_train)
    mean_r2_train = np.mean(all_r2_train)

    return mean_rmse_train, mean_r2_train, mean_rmse_test, mean_r2_test, all_fs, all_best_features, all_coefs, all_y_predict_train, all_y_predict_test, all_y_test, all_y_train, all_r2_test, all_r2_train, model, reg

def do_regression_split_given(X_train, y_train, X_test, y_test, what_sort, alpha, l1_ratio=0.5, do_PCA=False, kernel='rbf'):
    # same as do_regression, but we have a given train and testing set

    dict_to_return = {}
    dict_to_return['X_train'] = X_train
    dict_to_return['y_train'] = y_train
    dict_to_return['X_test'] = X_test
    dict_to_return['y_test'] = y_test
    
    random_state = randrange(100)
  
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    if what_sort=='SimpleRegression':
        model = LinearRegression()
    elif what_sort=='Lasso':
        model = Lasso(alpha=alpha, random_state=random_state)
    elif what_sort=='ElasticNet':
        model = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=random_state)
    elif what_sort=='Ridge':
        model = Ridge(alpha=alpha, random_state=random_state)
    elif what_sort=='SVM':
        model = svm.SVR(kernel=kernel, C=alpha)
    reg = model.fit(X_train, y_train)
    dict_to_return['reg'] = reg
    # evaluate the model
    y_predict_test = model.predict(X_test)
    y_predict_train = model.predict(X_train)
    
    dict_to_return['y_predict_test'] = y_predict_test
    dict_to_return['y_predict_train'] = y_predict_train
            
    # evaluate predictions
        
    dict_to_return['r2_test'] = reg.score(X_test,y_test)
    dict_to_return['r2_train'] = reg.score(X_train,y_train)


    dict_to_return['rmse_test'] = mean_squared_error(y_test, y_predict_test, squared=False)
    dict_to_return['rmse_train'] = mean_squared_error(y_train, y_predict_train, squared=False)
        
    if what_sort=='SVM':
        dict_to_return['all_coefs'] = reg.support_
    else:
        dict_to_return['all_coefs'] = reg.coef_
    

    return dict_to_return



def do_RandomForest_regression(X,y, n_splits, n_estimators, max_features, max_depth, min_samples_leaf):
    # Deal with overfitting:
        
    # n_estimators: in general the more trees the less likely the algorithm is to overfit. So try increasing this. The lower this number, the closer the model is to a decision tree, with a restricted feature set.
    # max_features: try reducing this number (try 30-50% of the number of features). This determines how many features each tree is randomly assigned. The smaller, the less likely to overfit, but too small will start to introduce under fitting. Fraction, default=1.0. if 0.5 it will take 50% of the features
    # max_depth: This will reduce the complexity of the learned models, lowering over fitting risk. Try starting small, say 5-10, and increasing you get the best result.
    # min_samples_leaf: Try setting this to values greater than one. This has a similar effect to the max_depth parameter, it means the branch will stop splitting once the leaves have that number of samples each.

 
    all_rmse_test = []
    all_rmse_train = []
    all_r2_test = []
    all_r2_train = []
    all_y_predict_test = []
    all_y_predict_train = []
    all_y_predict_test_proba = []
    all_y_predict_train_proba = []
    all_y_test = []
    all_y_train = []
    all_coefs = {}
    all_features = {}
    all_fs = {}
    index_dictionaries = 0
    
    kf = KFold(n_splits)
    
    for  train_index, test_index in kf.split(X,y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        random_state = randrange(100)
        model = RandomForestRegressor(random_state=random_state, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, n_jobs=4, min_samples_leaf=min_samples_leaf) # n_estimators=100, criterion='gini'
        
        reg = model.fit(X_train, y_train)

        # evaluate the model
        y_predict_test = model.predict(X_test)
        y_predict_train = model.predict(X_train)
       
        all_features[index_dictionaries] = reg.feature_importances_

        # evaluate predictions
        rmse_test = mean_squared_error(y_test, y_predict_test, squared=False)
        rmse_train = mean_squared_error(y_train, y_predict_train, squared=False)
        all_rmse_test.append(rmse_test)
        all_rmse_train.append(rmse_train)
        all_y_predict_test.append(y_predict_test)
        all_y_predict_train.append(y_predict_train)
        all_y_test.append(y_test)
        all_y_train.append(y_train)
        
        r2_test = r2_score(y_test.values, y_predict_test)
        r2_train = r2_score(y_train.values, y_predict_train)
        all_r2_test.append(r2_test)
        all_r2_train.append(r2_train)
       
        all_fs[index_dictionaries] = reg.feature_importances_
        all_coefs[index_dictionaries] =reg.feature_importances_
        index_dictionaries += 1
       
        index_dictionaries += 1
    mean_rmse_test = np.mean(all_rmse_test)
    mean_r2_test = np.mean(all_r2_test)
    mean_rmse_train = np.mean(all_rmse_train)
    mean_r2_train = np.mean(all_r2_train)

    return mean_rmse_train, mean_r2_train, mean_rmse_test, mean_r2_test, all_coefs, all_y_predict_train, all_y_predict_test, all_y_test, all_y_train, all_y_predict_test_proba, all_y_predict_train_proba, all_features, model, reg


def do_RandomForest_regression_split_given(X_train, y_train, X_test, y_test, n_estimators, max_features, max_depth, min_samples_leaf):
    # same as do_RandomForest_regression, but we have a given train and testing set
    dict_to_return = {}
    
    random_state = randrange(100)
    model = RandomForestRegressor(random_state=random_state, n_estimators=n_estimators, max_features=max_features, max_depth=max_depth, n_jobs=4, min_samples_leaf=min_samples_leaf) # n_estimators=100, criterion='gini'
        
    reg = model.fit(X_train, y_train)

    # evaluate the model
    y_predict_test = model.predict(X_test)
    y_predict_train = model.predict(X_train)
    
    dict_to_return['y_predict_test'] = y_predict_test
    dict_to_return['y_predict_train'] = y_predict_train
       
    dict_to_return['all_coefs'] = reg.feature_importances_

    # evaluate predictions
    dict_to_return['RMSE_test'] = mean_squared_error(y_test, y_predict_test, squared=False)
    dict_to_return['RMSE_train'] = mean_squared_error(y_train, y_predict_train, squared=False)
            
    dict_to_return['r2_test'] = r2_score(y_test.values, y_predict_test)
    dict_to_return['r2_train'] = r2_score(y_train.values, y_predict_train)
    
    dict_to_return['y_test'] = y_test
    dict_to_return['y_train'] = y_train
    dict_to_return['X_train'] = X_train
    dict_to_return['X_test'] = X_test

    dict_to_return['reg'] = reg

    return dict_to_return


#*******************************************************************************
#**************************** LINEAR REGRESSION *********************************
#*******************************************************************************

def RMSE_regression(all_data_features, to_predict, n_split, what_sort, alpha, l1_ratio=0.5, do_PCA=False, kernel='rbf'):#, path_out):
     
    # alphafloat, default=1.0
    # Constant that multiplies the L1 term. Defaults to 1.0. alpha = 0 is equivalent 
    # to an ordinary least square, solved by the LinearRegression object. 
    # For numerical reasons, using alpha = 0 with the Lasso object is not advised. 
    # Given this, you should use the LinearRegression object.

    
    
    all_features = list(all_data_features.columns)

    X = all_data_features
    X = X.apply(pd.to_numeric)
    y = to_predict
    y = y.apply(pd.to_numeric)
 
    number_best_features = int(len(X.columns)/5)
    if not number_best_features:
        number_best_features =1
        
    
    RMSE_train, r2_train, RMSE_test, r2_test, fs, best_features, all_coefs, all_y_predict_train, all_y_predict_test, all_y_test, all_y_train, all_r2_test, all_r2_train, model, reg = do_regression(X=X,y=y, number_best_features=number_best_features, n_splits=n_split, what_sort=what_sort, alpha=alpha, l1_ratio=l1_ratio, do_PCA=do_PCA, kernel=kernel)
     
    dict_to_return = {}
    dict_to_return['RMSE_train'] = RMSE_train
    dict_to_return['RMSE_test'] = RMSE_test
    dict_to_return['r2_train'] = r2_train
    dict_to_return['r2_test'] = r2_test
    dict_to_return['fs'] = fs
    dict_to_return['best_features'] = best_features
    dict_to_return['all_coefs'] = all_coefs
    dict_to_return['all_y_predict_train'] = all_y_predict_train
    dict_to_return['all_y_predict_test'] = all_y_predict_test
    dict_to_return['all_y_test'] = all_y_test
    dict_to_return['all_y_train'] = all_y_train
    dict_to_return['all_r2_test'] = all_r2_test
    dict_to_return['all_r2_train'] = all_r2_train
    dict_to_return['model'] = model
    dict_to_return['reg'] = reg
    
    return dict_to_return

def Forest_regression(X, 
                      y, 
                      n_split=10, 
                      n_estimators=100, 
                      max_features="auto", 
                      max_depth=None, 
                      min_samples_leaf=1):#, path_out):
   # - n_estimatorsint, default=100:      The number of trees in the forest.
   # - max_features{“auto”, “sqrt”, “log2”}, int or float, default=”auto”: 
       # The number of features to consider when looking for the best split:
        # If int, then consider max_features features at each split.
        # If float, then max_features is a fraction and round(max_features * n_features) features are considered at each split.
    
  # max_depth: default None
    
    mean_rmse_train, mean_r2_train, mean_rmse_test, mean_r2_test, all_coefs, all_y_predict_train, all_y_predict_test, all_y_test, all_y_train, all_y_predict_test_proba, all_y_predict_train_proba, all_features, model, reg = do_RandomForest_regression(X,y, n_split, n_estimators, max_features, max_depth,min_samples_leaf)
    
    
    dict_to_return = {}
    dict_to_return['RMSE_train'] = mean_rmse_train
    dict_to_return['RMSE_test'] = mean_rmse_test
    dict_to_return['r2_train'] = mean_r2_train
    dict_to_return['r2_test'] = mean_r2_test
    dict_to_return['all_coefs'] = all_coefs
    dict_to_return['all_y_predict_train'] = all_y_predict_train
    dict_to_return['all_y_predict_test'] = all_y_predict_test
    dict_to_return['all_y_test'] = all_y_test
    dict_to_return['all_y_train'] = all_y_train
    dict_to_return['model'] = model
    dict_to_return['reg'] = reg
    
    return dict_to_return

