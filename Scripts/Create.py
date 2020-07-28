# -*- coding: utf-8 -*-
"""
Created on Thu Jun 25 16:37:00 2020

@author: Andrés PC
"""

# Creación nuevas variables.

import numpy as np
directorio = "C:\proyecto final\datos"
import os
cwd = os.getcwd()
os.chdir(directorio)
os.listdir()

import pandas as pd
import matplotlib.pyplot as plt





def create1(datos):
    datos['Partida'] = np.vectorize(location)(datos['pickup_latitude'], datos['pickup_longitude'])
    datos['Destino'] = np.vectorize(location)(datos['dropoff_latitude'], datos['dropoff_longitude']) 
    
    datos.loc[datos['Partida']=='None', 'Partida']= 'NEW JERSEY'
    datos.loc[datos['Destino']=='None', 'Destino']= 'NEW JERSEY' 
    
    datos['distancia_geo'] = datos.apply(lambda x: distancia(x), axis = 1)
    datos['distancia_ciudad'] = (abs(datos.dropoff_longitude - datos.pickup_longitude) +
                            abs(datos.dropoff_latitude - datos.pickup_latitude)) 
    
    mean_dist = datos ['distancia_geo']. mean () 
    datos.loc [datos ['distancia_geo'] == 0, 'distancia_geo'] = mean_dist

    mean_dist2 = datos ['distancia_ciudad']. mean () 
    datos.loc [datos ['distancia_ciudad'] == 0, 'distancia_ciudad'] = mean_dist2
    
    datos['Partida']= datos['Partida'].astype(str)
    datos['Destino']= datos['Destino'].astype(str)
    
    return datos



def create2(datos):
    datos['conca']= datos['Partida'] + datos['Destino']
    datos['frecuencia_viajes'] = datos.groupby('conca')['conca'].transform('count')
    datos['media_duration']= datos['frecuencia_viajes']
    
    # Mhanhatan 
    datos['media_duration'] = datos['media_duration'].replace(567939,6.228472)
    datos['media_duration'] = datos['media_duration'].replace(1195,6.725941)
    datos['media_duration'] = datos['media_duration'].replace(146697,6.725941)
    datos['media_duration'] = datos['media_duration'].replace(138007,6.725941)
    datos['media_duration'] = datos['media_duration'].replace(94,6.725941)

    # Bronx
    datos['media_duration'] = datos['media_duration'].replace(103,5.313066)
    datos['media_duration'] = datos['media_duration'].replace(2,6.725941)
    datos['media_duration'] = datos['media_duration'].replace(41,6.725941)
    datos['media_duration'] = datos['media_duration'].replace(18,6.725941)
    
    # #Media BROOKLYN
    datos['media_duration'] = datos['media_duration'].replace(275,6.79069)
    datos['media_duration'] = datos['media_duration'].replace(161657,6.177003)
    datos['media_duration'] = datos['media_duration'].replace(138335,6.79069)
    datos['media_duration'] = datos['media_duration'].replace(29,6.79069)
    datos['media_duration'] = datos['media_duration'].replace(40197,6.79069)
    
    # #Media Queens
    datos['media_duration'] = datos['media_duration'].replace(772,6.949564)
    datos['media_duration'] = datos['media_duration'].replace(29262,6.949564)
    datos['media_duration'] = datos['media_duration'].replace(118302,6.949564)
    datos['media_duration'] = datos['media_duration'].replace(28,6.949564)
    datos['media_duration'] = datos['media_duration'].replace(95557,6.089501)
    
    # New Jersey
    datos['media_duration'] = datos['media_duration'].replace(7,6.779935)
    datos['media_duration'] = datos['media_duration'].replace(54,3.976765)
    
    # Creación tipo servicio
    datos['tipo_servicio'] = np.vectorize(tipo_viaje)(datos['Partida'], datos['Destino'])
    datos['media_global_duration']= datos['tipo_servicio']
    datos['media_global_duration'] = datos['media_global_duration'].replace('interno',5.556961)
    datos['media_global_duration'] = datos['media_global_duration'].replace('externo',6.813698)
    
    datos['tipo_viaje']= datos['media_duration']
    datos['tipo_viaje'][datos.media_duration == 6.089501] = 'viaje_medio_inter'
    datos['tipo_viaje'][datos.media_duration == 6.79069] = 'viaje_medio_exter'
    datos['tipo_viaje'][datos.media_duration== 5.313066] = 'viaje_medio_inter'
    datos['tipo_viaje'][datos.media_duration== 3.976765] = 'viaje_medio_inter'
    datos['tipo_viaje'][datos.media_duration== 6.725941 ] = 'viaje_corto_exter'
    datos['tipo_viaje'][datos.media_duration== 6.779935 ] = 'viaje_corto_exter'
    datos['tipo_viaje'][datos.media_duration== 6.228472 ] = 'viaje_largo_inter'
    datos['tipo_viaje'][datos.media_duration== 6.177003 ] = 'viaje_largo_inter'
    # datos['tipo_viaje'][datos.media_duration== 6.759935 ] = 'viaje_corto_exter'
    datos['tipo_viaje'][datos.media_duration >= 6.81414] = 'viaje_medio_exter'
    
    datos.drop(datos[['conca','frecuencia_viajes']],axis=1, inplace =True)
    
    datos['tipo_viaje']= datos['tipo_viaje'].astype(str)
    datos['tipo_servicio']= datos['tipo_servicio'].astype(str)
    return datos
    
    


    
    
    
    
    
    
    






























# df['media'].value_counts()












# internos externos.








