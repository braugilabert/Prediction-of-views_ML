import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn

data = pd.read_csv('Prediction_views_ML/data/raw/dataset siralatriste.csv')

#para sacar la media de compartidos de cada uno de mis juegos 
data['Med Juego Compartido'] = data.groupby('Juego')['Compartido'].transform('mean')

#para sacar la media de comentarios de cada uno de mis juegos 
data['Med Juego Comentarios'] = data.groupby('Juego')['Comentarios'].transform('mean')

#para sacar la media del porcentaje de likes/dislikes de cada uno de mis juegos
data['Med porcentaje likes/dislikes Juego'] = data.groupby('Juego')['Me gusta (vs. No me gusta) (%)'].transform('mean')

#para sacar la media de likes de cada uno de mis juegos
data['Med Likes x juego'] = data.groupby('Juego')['Me gusta'].transform('mean')

#para sacar la media de tiempo (en horas) de visualizacion de cada uno de mis juegos
data['Med Horas vistas Juego'] = data.groupby('Juego')['Tiempo de visualización (horas)'].transform('mean')

#para sacar la duracion media de (en minutos) de las visualizaciones hechas en cada uno de mis juegos
data['Med Duracion de Visualizaciones Juego'] = data.groupby('Juego')['Duración media de las visualizaciones'].transform('mean')

#para sacar la media de impresiones para cada uno de mis juegos
data['Med Impresiones Juego'] = data.groupby('Juego')['Impresiones'].transform('mean')

#para sacar la media de porcentaje de click de impresiones para cada uno de mis juegos
data['Med porcentaje clicks Impresiones Juego'] = data.groupby('Juego')['Porcentaje de clics de las impresiones (%)'].transform('mean')

#para sacar la media de visualizaciones para cada uno de mis juegos
data['Med Visualizaciones Juego'] = data.groupby('Juego')['Visualizaciones'].transform('mean')

#para sacar la media de suscriptores ganados en cada uno de los juegos
data['Med Suscriptores Juego'] = data.groupby('Juego')['Suscriptores'].transform('mean')

from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()
data['Juego_int'] = le.fit_transform(data['Juego'])

data['Título_1'] = np.where(data['Título del vídeo'].str.contains("1"), 1 , 0) #PARA CUANDO UNA COLUMNA CONTIENE CIERTA PALABRA
#data[data['Título_1']==1]['Título del vídeo'].value_counts(normalize=True) #para contar dichos valores 1 o 0

data['Año'] = data['Fecha'].str.split().str[-1].astype('int') #para coger solo la última palabra y hacerlo numero entero
#data.head(1)

data['Historia'] = np.where(data['Juego'].str.contains('Rome 2 Total War|Empire Total War|Napoleón Total War|Imperivm 3|Europa Universalis 4|Medieval Total War 2|Naval Action', regex=True), 1 , 0) #PARA CUANDO UNA COLUMNA CONTIENE CIERTA PALABRA

data['TW'] = np.where(data['Juego'].str.contains('Total War'), 1 , 0)

#data[data['TW']==1]['Juego'].value_counts(normalize=True) #para contar dichos valores 1 


data['Med Visualizaciones Año'] = data.groupby('Año')['Visualizaciones'].transform('mean')

data['Visualizaciones necesarias para suscribirse al canal'] = data['Visualizaciones'] / data['Suscriptores']

data['Visualizaciones necesarias para compartir el video'] = data['Visualizaciones'] / data['Compartido']

# PARA SUSTITUIR LOS inf de las últimas dos variables por 0
data['Visualizaciones necesarias para suscribirse al canal'].replace([np.inf, -np.inf], 0, inplace=True)
data['Visualizaciones necesarias para compartir el video'].replace([np.inf, -np.inf], 0, inplace=True)

# Eliminacion de columnas

data.drop(columns=['Compartido', 'Tiempo de visualización (horas)', 'Suscriptores', 'Impresiones', 'Me gusta', 'Comentarios'], inplace=True)

data.drop(columns=['SEO', 'Año', 'Juego_int', 'Título_1', 'Med Duracion de Visualizaciones Juego', 'Me gusta (vs. No me gusta) (%)', 'Med Likes x juego', 'TW', 'Duración media de las visualizaciones', 'Med porcentaje likes/dislikes Juego', 'Historia', 'Med Juego Comentarios', 'Fecha', 'Juego', 'Datos Duración media de las visualizaciones'], inplace=True)

#GUARDAR ARCHIVO
data.to_csv('Prediction_views_ML/data/processed.csv', index=False)
