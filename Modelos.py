# -*- coding: utf-8 -*-
"""
Created on Sun May 24 18:44:22 2020

@author: Andrés PC
"""

# Regresión lineal

import math
import pandas as pd
import numpy as np
import keras.backend as K
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
import random
import re
from joblib import dump
from tqdm import tqdm
from mlxtend.plotting import plot_learning_curves



from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.externals import joblib 

from keras.datasets import imdb
from keras.wrappers.scikit_learn import KerasClassifier
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras import models
from keras import layers
from keras import regularizers
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU


from sklearn import linear_model
import xgboost as xgb
import lightgbm as lgb



def regresion_lineal( x,y,state,size,title,title2,title3):
  "Esta función crea un modelo lineal a partir de  unos datos previamente cargados y depurados. Para ello le damos nuestras variables dependiente y independiente,el random state y el tamaño del test para la creación del conjunto de test y train"
  
  #Creacion conjunto de train y test
  X_train, X_test, Y_train, Y_test = train_test_split(x, y,random_state =state, test_size = size)
  
  # Creación del objecto lineal y entrenamiento
  reg = linear_model.LinearRegression()
  reg.fit(X_train, Y_train)
  
  # prediciendo sobre el conjunto de datos de entrenamiento
  y_train_predicted = reg.predict(X_train)
  
  # prediciendo sobre el conjunto de datos de test
  y_test_predict = reg.predict(X_test)
  
  # Evaluando el modelo con los datos de entrenamiento
  rmse_train = (np.sqrt(mean_squared_error(Y_train, y_train_predicted)))
  # rmsle_train =(np.sqrt(mean_squared_log_error( Y_train, y_train_predicted)))
  r2_train = r2_score(Y_train, y_train_predicted)
  
  # Evaluando el modelo con los datos test o de la realidad
  rmse_test = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
  # rmsle_test =(np.sqrt(mean_squared_log_error( Y_test, y_test_predict )))
  r2_test = r2_score(Y_test, y_test_predict)
  
  # print('Gráfico sus datos antes aplicar el modelo del modelo')
  # plt.scatter(x, y)
  # plt.title(title)
  # plt.xlabel(title2)
  # plt.ylabel(title3)
  # plt.show()
  print("-------------------------------------------")

  print("El rendimiento del modelo para el conjunto de train es")
  print("-------------------------------------------")
  print("RMSE del set de entrenamiento es {}".format(rmse_train))
  # print("RMSLE score del set de entreamiento es {}".format(rmsle_train))
  # print("R2 score del set de entreamiento es {}".format(r2_train))
  
  print('Gráfico conjunto de train')
  plt.scatter(Y_train, y_train_predicted)
  plt.title(title)
  plt.xlabel(title2)
  plt.ylabel(title3)
  #plt.plot(X_train, y_train_predicted, "r-", linewidth=3)
  plt.show()
  
  print("\n")
  
  print("El rendimiento del modelo para el conjunto de test es")
  print("-------------------------------------------")
  print("RMSE del set de test es {}".format((rmse_test)))
  # print("RMSLE del set de test es {}".format(rmsle_test))
  # print("R2 score del set de test es  {}".format(r2_test))
  
  print('Gráfico conjunto de test')
  plt.scatter(Y_test, y_test_predict)
  plt.title(title)
  plt.xlabel(title2)
  plt.ylabel(title3)
  # plt.plot(X_test, y_test_predict, "r-", linewidth=3)
  plt.show()
  


#-----------------------------------------------------------------------------
# modelo XGBOOST

def xgboost(X,y):
    xgb_pars = {
    'boosting_type': ['gbdt'],
    'objective': ['reg:squarederror'],
    'metric': ['rmse'],
    'colsample_bylevel':[ 1.0],
    'max_depth': [6], 
    'learning_rate':[ 0.1],
    'verbose': [0], 
    'min_child_weight': [0.001],
    'early_stopping_round': [10],
    'n_estimators': [600],
    'colsample_bytree':[0.7],
    'reg_lambda': [0.0],
    'nthread': [-1],
    'reg_alpha': [0.2],
    'subsample':[1.0]}

    model = xgb.XGBRegressor()
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
    
    grid_solver = GridSearchCV(estimator = model, 
                   param_grid = xgb_pars,
                   scoring='neg_mean_squared_error',
                   cv = 5,
                   n_jobs=-1,
                   verbose = 2)
    
    model_result_xgboost = grid_solver.fit(X_train,y_train)
    y_pred=model_result_xgboost.predict(X_train)
    print("RMSE:", np.sqrt(mean_squared_error(y_train, y_pred)))
  
    y_pred_test =model_result_xgboost.predict(X_test)
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
    
    f = plt.figure(figsize=(10,8))
    plt.title('Feature importances')
    plt.ylabel("Feature")
    plt.xlabel("Relative Importance")(pd.Series(model_result_xgboost.best_estimator_.feature_importances_, index=X.columns)
                                      .nlargest(10)
                                      .plot(kind='barh'))

    
   
    
    
def lightgbm(X,y):
    X = X.rename(columns = lambda x:re.sub('[^A-Za-z0-9_]+', '', x))
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.25, random_state=15)
    
    gbm = lgb.LGBMRegressor()
    params = {'boosting_type': ['gbdt'],
 'class_weight':[ None],
 'colsample_bytree':[ 1.0], 
 'importance_type': ['split'],
 'learning_rate':[ 0.1],
 'max_depth': [-1],
 'min_child_samples':[ 20],
 'min_child_weight': [0.001],
 'min_split_gain': [0.0],
 'n_estimators': [600],
 'n_jobs': [-1],
 'metric': ['rmse'],
 'num_leaves': [50],
 'objective': ['regression'],
 'random_state': [None],
 'reg_alpha': [0.2],
 'reg_lambda': [0.0],
 'silent': [1],
 'early_stopping_rounds':[10],
 'subsample': [1.0],
 'subsample_for_bin':[ 200000],
 'subsample_freq': [0]}
    
    grid_solver = GridSearchCV(estimator = gbm, # model to train
                   param_grid = params,
                   scoring='neg_mean_squared_error',
                   cv = 5,
                   n_jobs=-1,
                   verbose = 2)
    
    model_result_LightGBM = grid_solver.fit(X_train,y_train, eval_set = (X_test, y_test))
    
    y_pred_train=model_result_LightGBM.predict(X_train)
    print("RMSE: ",  np.sqrt(mean_squared_error(y_train, y_pred_train)))
    
    y_pred_test =model_result_LightGBM.predict(X_test)
    print("RMSE:", np.sqrt(mean_squared_error(y_test, y_pred_test)))
    
    f = plt.figure(figsize=(10,8))
    plt.title('Feature importances')
    plt.ylabel("Feature")
    plt.xlabel("Relative Importance")(pd.Series(model_result_LightGBM.best_estimator_.feature_importances_, index=X.columns)
                                      .nlargest(10)
                                      .plot(kind='barh'))
    
 
def NLP(X,y,epochs): 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=15)
    model = Sequential()
    model.add(Dense(13, input_dim=33, kernel_initializer='normal', activation='relu'))
    model.add(BatchNormalization())
    model.add(Dense(512,activation='relu'))#512 neurons in input layer
    model.add(Dropout(0.2))
    model.add(BatchNormalization())
    model.add(Dense(256,activation='relu')) #256 neurons in hidden layer
    model.add(BatchNormalization())
    model.add(Dense(128,activation='relu'))  # 128 neurons in hidden layer
    model.add(BatchNormalization())
    model.add(Dense(64,activation='relu'))   # 64 neurons in hidden layer
    model.add(BatchNormalization())
    model.add(Dense(32,activation='relu'))   # 32 neurons in hidden layer
    model.add(BatchNormalization())
    model.add(Dense(16,activation='relu')) # 16 neurons in hidden layer
    model.add(BatchNormalization())
    model.add(Dense(8,activation='relu')) # 8 neurons in hidden layer
    model.add(BatchNormalization())
    model.add(layers.Dense(1, activation='sigmoid'))
    
    model.compile(optimizer='adam',loss='mean_squared_error', metrics=[tf.keras.metrics.MeanSquaredError()])
    model.summary()
    
    history = model.fit(X_train,
                    y_train,
                    epochs=epochs,
                    validation_data=(X_test, y_test))
    
    test_loss, test_acc = model.evaluate(X_test, y_test)
    print('test_RMSE:', test_acc)
  



    


    







