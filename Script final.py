# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 12:23:18 2020

@author: Andrés PC
"""


directorio = "C:\proyecto final\datos"
import os
cwd = os.getcwd()
os.chdir(directorio)
os.listdir()

# Librerias que utilizaremos 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Cargamos nuestros modulos con las funciones utilizdas
exec(open("Cleaning.py").read())
exec(open("Create.py").read())
exec(open("Plots.py").read())
exec(open("Modelos.py").read())
exec(open("Funciones utilizadas.py").read())


pd.describe_option('display')  
pd.set_option('display.max_rows', 445) 
pd.set_option('display.max_columns',20) 

datos = pd.read_csv('datos_taxi.csv', delimiter = ",")
datos = pd.DataFrame(datos)

# # In this step we will apply the cleaning funcion to our data for deleiting the outlier values of data 
datos = cleaning(datos)

# Apply the create funcion:
# these funcions generate news columns from the above information and return our
# final dataset
datos = create1(datos)
datos = create2(datos)

# Finally we will apply the funcion that contents our models (lineal reggresion, XGBOOTS, LIGTHGBM, LBM)

# Separeted  our target from of final dataset
y = datos['trip_duration']

# Deleted it
datos.drop(datos[['trip_duration']], axis = 1, inplace = True)

# Convert to dammies
X = pd.get_dummies(datos)

# Apply the lineal regression model
regresion_lineal(X,y,15,0.25,title='título gráfico',title2= 'título eje x', title3 ='título eje y')

# Apply XGBOOST
xgboost(X,y)

# Apply LIGTHGBM
lightgbm(X,y)

# Apply Deep Learning
NLP(X,y,epochs=20)



