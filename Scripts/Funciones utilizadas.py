# -*- coding: utf-8 -*-
"""
Created on Wed Jun 17 08:58:08 2020

@author: Andrés PC
"""

# Funciones utilizadas en el proyecto:
    
# para encontrar la distancia por coordenadas
def distancia(df):  
    import numpy as np
    # convertimos a radianes
    lat1, lon1, lat2, lon2 = df.pickup_latitude,df.pickup_longitude,df.dropoff_latitude,df.dropoff_longitude
    lat1, lon1, lat2, lon2 = map(np.radians, (lat1, lon1, lat2, lon2))
    R = 6373 
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = np.sin(dlat / 2)**2 +np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2)**2
    distancia = 2 * R* np.arcsin(np.sqrt( a))
    return distancia


from math import radians, cos, sin, asin, sqrt
def haversine(df):
    lat1, lon1, lat2, lon2 = df.pickup_latitude,df.pickup_longitude,df.dropoff_latitude,df.dropoff_longitude 
    R = 3959.87433 # this is in miles.  For Earth radius in kilometers use 6372.8 km
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))
    return R * c

 # datos['distance'] = datos.apply(lambda x: haversine(x), axis = 1)


def distancia_ciudad(lat1, lng1, lat2, lng2):
    a = distancia(lat1, lng1, lat1, lng2)
    b = distancia(lat1, lng1, lat2, lng1)
    return a + b 

def distancia_ciudad2(df):
    lat1, lon1, lat2, lon2 = df.pickup_latitude,df.pickup_longitude,df.dropoff_latitude,df.dropoff_longitude
    a = distancia(df)
    b = distancia(df)
    return a + b 


# Para un ánalisis más descriptivo de cada variable
def values(df):
    it=df.columns
    for col in it:
        vle=pd.DataFrame(df[col].value_counts(dropna=False))
        vle['%']=((vle[col]/(vle[col].sum()))*100)
        vle.loc['Total'] = vle.sum()
        vle['%']=vle['%'].apply(lambda x:round(x,1))
        vle[col]=vle[col].astype(int)
        print(vle,'\nNum. Categories:',vle.shape[0]-1,'\n')
    return

# values('nombre dataframe'[['columas del data frame']])

# Para el análisis de los valores nulos.
def missings(df,title):
    import seaborn as sns
    import matplotlib.pyplot as plt
    total = df.isnull().sum().sort_values(ascending=False)
    percent = (df.isnull().sum()*100/
    df.isnull().count()).sort_values(ascending=False)
    missing_data = pd.concat([total, percent], axis=1, keys=['Total', 'Porcentaje'])
    print(missing_data.head(20))
    sns.heatmap(df.isnull(),yticklabels=False,cbar={'label':'Density'},cmap='Blues')
    plt.suptitle(title)
    
# missings(house_price,title='Distribución valores nulos')

# Para ver la correlación entre variables.
def columnas_correlacionadas(df):
    "La siguiente función lo que hace mostranos los pares de varibles que estanmás correlacioandas entre ellas."
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    pairs_correlated = [(row,column,upper.loc[row,column]) for column in upper.columns for  row in upper.columns if upper.loc[row,column] > 0.6]
    return pairs_correlated


# Correlación de varialbes con el target con el target.
def correlation_target(df,k,target):
    import seaborn as sns
    import matplotlib.pyplot as plt
    corrmat = df.corr()
    k = k
    cols = corrmat.nlargest(k, target)[target].index 
    cm = np.corrcoef(df[cols].values.T)
    sns.set(font_scale=1.25)
    hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f',    
    annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
    plt.show()          
                     
# correlation_target(titanic,10,target='Survived')    


def location(lat, long):
    if 40.498819681472185 <= lat < 40.647980383082 and -74.25127710072611 <= long < -74.05871817021664:
        return 'STATEN ISLAND'
    elif 40.56397537161681<= lat < 40.79952438531463 and -73.96069664354937 <= long < -73.70230256557906:
        return 'QUEENS'
    elif 40.57200646899509 <= lat < 40.738817781235454 and -74.04081093578047 <= long < -73.85646197859525: 
        return 'BROOKLYN'
    elif 40.69279306033091 <= lat < 40.877449669903285 and -74.01827562466266 <= long < -73.90788595913352: 
        return 'MANHATTAN'
    elif 40.79846803456301 <= lat < 40.91217840308113 and -73.93144726517058 <= long < -73.78287248508497:
        return 'BRONX'

def tipo_viaje(partida, destino):
    if partida == destino:
        return 'interno'
    else:
        return 'externo'

def create_coordenadas(latitud_partida,latitud_destino,longitud_partida,longitud_destino):
    import pandas as pd
    longitude = list(longitud_partida) + list(longitud_destino)
    latitude = list(latitud_partida) + list(latitud_destino)
    coordenadas = pd.DataFrame()
    coordenadas['longitud'] = longitude
    coordenadas['latitud'] = latitude
    return coordenadas

    

