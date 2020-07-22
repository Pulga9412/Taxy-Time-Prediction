# -*- coding: utf-8 -*-
"""
Created on Sun Jun 21 20:53:57 2020

@author: Andrés PC
"""

import numpy as np
directorio = "C:\proyecto final\datos"
import os
cwd = os.getcwd()
os.chdir(directorio)
os.listdir()

import pandas as pd
import matplotlib.pyplot as plt
pd.describe_option('display')  
pd.set_option('display.max_rows', 445) 
pd.set_option('display.max_columns',20) 


# Cargamos  nuestro data se a trabajar.
datos = pd.read_csv('datos_taxi.csv', delimiter = ",")
datos = pd.DataFrame(datos)


def cleaning(datos):
    datos['vendor_id']= datos['vendor_id'].astype(str)
    datos.drop(datos[datos.passenger_count== 0].index,inplace =True)
    datos.loc[datos['passenger_count']>=3, 'passenger_count']= 'Más de 2'
    datos.loc[datos['passenger_count']==1, 'passenger_count']= 'Entre 1 y 2'
    datos.loc[datos['passenger_count']==2, 'passenger_count']= 'Entre 1 y 2'
    datos.drop(datos[['store_and_fwd_flag']], axis=1, inplace= True)
    
    datos[['pickup_datetime']] = pd.to_datetime(datos['pickup_datetime'])
    datos['dia_semana']= datos['pickup_datetime'].dt.weekday
    datos['hora']= datos['pickup_datetime'].dt.hour

    datos.loc[datos['hora']>=17, 'hora']= 'Noche'
    datos.loc[datos['hora']==0, 'hora']= 'Noche'
    datos.loc[datos['hora']==12, 'hora']= 'Tarde'
    datos.loc[datos['hora']==18, 'hora']= 'Tarde'
    datos.loc[datos['hora']==17, 'hora']= 'Tarde'
    datos.loc[datos['hora']==14, 'hora']= 'Tarde'
    datos.loc[datos['hora']==13, 'hora']= 'Tarde'
    datos.loc[datos['hora']==15, 'hora']= 'Tarde'
    datos.loc[datos['hora']==16, 'hora']= 'Tarde'
    datos.loc[datos['hora']==11, 'hora']= 'Tarde'
    datos.loc[datos['hora']==10, 'hora']= 'Mañana'
    datos.loc[datos['hora']==9, 'hora']= 'Mañana'
    datos.loc[datos['hora']==8, 'hora']= 'Mañana'
    datos.loc[datos['hora']==6, 'hora']= 'Madrugada'
    datos.loc[datos['hora']==1, 'hora']= 'Madrugada'
    datos.loc[datos['hora']==2, 'hora']= 'Madrugada'
    datos.loc[datos['hora']==3, 'hora']= 'Madrugada'
    datos.loc[datos['hora']==4, 'hora']= 'Madrugada'
    datos.loc[datos['hora']==5, 'hora']= 'Madrugada'
    datos.loc[datos['hora']==7, 'hora']= 'Madrugada'
      
    datos.drop(['pickup_datetime'], axis =1, inplace = True)
    datos.drop(['dropoff_datetime'], axis =1, inplace = True)
      
    xlim = [-74.03, -73.77]
    ylim = [40.63, 40.85]
    datos = datos[(datos.pickup_longitude> xlim[0]) & (datos.pickup_longitude < xlim[1])]
    datos = datos[(datos.dropoff_longitude> xlim[0]) & (datos.dropoff_longitude < xlim[1])]
    datos = datos[(datos.pickup_latitude> ylim[0]) & (datos.pickup_latitude < ylim[1])]
    datos= datos[(datos.dropoff_latitude> ylim[0]) & (datos.dropoff_latitude < ylim[1])]
      
    datos['trip_duration']= np.log(datos['trip_duration']+1)
    datos.drop(['extra'], axis =1, inplace = True)
    datos.drop(['total_price'], axis =1, inplace = True)
    datos.drop(['payment_type'], axis =1, inplace = True)
    datos.drop(['id'], axis =1, inplace = True)
 
  
    return datos
 


  



 




    
    
   
  



    


    

    
    
