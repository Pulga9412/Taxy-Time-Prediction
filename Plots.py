# -*- coding: utf-8 -*-
"""
Created on Sat Jun 20 13:48:18 2020

@author: Andrés PC
"""

# Análisis gráfico de las variables.


directorio = "C:\proyecto final\datos"
import os
cwd = os.getcwd()
os.chdir(directorio)
os.listdir()

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier

pd.describe_option('display')  
pd.set_option('display.max_rows', 445) 
pd.set_option('display.max_columns',20) 


# Cargamos  nuestro data se a trabajar.
datos = pd.read_csv('datos_taxi.csv', delimiter = ",")
datos = pd.DataFrame(datos)


# Varibales categoricas:
    
def plots_univariante(df, title1,title2):
    """ Esta función nos genera un plot univariante de la columna que le 
    bindemos"""
    f = plt.figure(figsize=(10,8))
    sns.countplot(x=df, data=df)
    plt.xlabel(title1, fontsize=14)
    plt.ylabel(title2, fontsize=14)
    plt.show()
    print('su gráfico es el siguiente')
    
# plots_univariantes(datos['vendor_id'],title1="",title2="")

    
def plots_bivaiantes(df,y, title1, title2):
    f = plt.figure(figsize=(10,8))
    sns.barplot(df, y)
    plt.xlabel(title1, fontsize=14)
    plt.ylabel(title2, fontsize=14)
    plt.legend(loc=(1.04,0))
    plt.show()
    print('su gráfico es el siguiente')


 # plots_bivaiantes(datos['vendor_id'], y=datos['trip_duration'],title1="",title2="")
    
# Variables continuas:

def plots_continua(df,title1,title2):
    f = plt.figure(figsize=(10,8))
    sns.distplot(df)
    plt.xlabel(title1, fontsize=14)
    plt.ylabel(title2, fontsize=14)
    plt.legend(loc=(1.04,0))
    plt.show()
    print('su gráfico es el siguiente')

# plots_continua(datos['pickup_latitude'],title1="",title2="")


def plots_trivariante(df,df1,name,y,title1,title2):
    f = plt.figure(figsize=(10,8))
    df.groupby([df1, name])[y].sum().unstack().plot()
    plt.xlabel(title1, fontsize=14)
    plt.ylabel(title2, fontsize=14)
    plt.show()
    
    
# plots_trivariante(datos, datos['vendor_id'], name='payment_type',y='trip_duration',title1="hola",title2="hola2")

def plot_pai(df,title):
    plt.figure(figsize=(10,8))
    df.value_counts().plot(kind='pie', autopct='%1.1f%%')
    plt.title(title)

 # plot_pai(datos['tipo_viaje'],title="Duración de los viajes")
    

    
def plot_headmapt(df,df1,name, y,title1,title2):
    mapa = df.groupby(by=[df1,name]).count()[y].unstack()
    plt.figure(figsize=(18,3))
    sns.set(font_scale=1)
    sns.heatmap(mapa,
            cmap='Blues',
            xticklabels=True,
            vmin=0,
            linewidths=0.5,
            linecolor='black',
            cbar_kws={'label': title1})
    plt.suptitle(title2)
    
 # plot_headmapt(datos,datos['vendor_id'],name ='payment_type', y = 'trip_duration', title1="Densidad", title2 = 'Promedio de')
    
    
def plot_continuo2(df,y):
    plt.figure(figsize=(16,5))
    for p in set(df):
        sns.distplot(y[df==p],
        hist=True,kde=False,label=p)
        plt.title("Distribution plot")
        plt.legend()

    
# plot_continuo2(datos['vendor_id'],y= datos['trip_duration']) 
        
def plot_mapa(latitud_partida,latitud_destino,longitud_partida,longitud_destino):
    longitude = list(longitud_partida) + list(longitud_destino)
    latitude = list(latitud_partida) + list(latitud_destino)
    plt.figure(figsize = (10,10))
    plt.plot(longitude,latitude,'.', alpha = 0.4, markersize = 0.05)
    plt.show()
    
# plot_mapa(latitud_partida = datos['pickup_latitude'],latitud_destino = datos['dropoff_latitude'], longitud_partida =datos['pickup_longitude'],longitud_destino = datos['dropoff_longitude'])



def plot_clusters(df1,df_label,df_longitud,df_latitud,cluster):
    kmeans = cluster
    fig,ax = plt.subplots(figsize = (10,10))
    for label in df1.label.unique():
        ax.plot(df_longitud[df_label == label],df_latitud[df_label == label],'.', alpha = 0.3, markersize = 0.3)
        ax.plot(kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1],'o', color = 'r')
        ax.annotate(label,(kmeans.cluster_centers_[label,0],kmeans.cluster_centers_[label,1]), color = 'black', fontsize = 20)
    ax.set_title('Gráfico de Clusters')
    plt.show()
        
# plot_clusters(datos1,datos1['label'],datos1['longitude'],datos1['latitude'],cluster = kmeans)


def feature_importances(model, df):
    """
    Plot feature importances of the model when you using gridsearch (model is the result of the gridSearch fit).
    """
    f = plt.figure(figsize=(10,8))
    plt.title('Feature importances')
    plt.ylabel("Feature")
    plt.xlabel("Relative Importance")
    (pd.Series(model.best_estimator_.feature_importances_, index=df.columns)
                                      .nlargest(10)
                                      .plot(kind='barh'))
    

