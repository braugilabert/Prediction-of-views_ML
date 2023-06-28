import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error


df = pd.read_csv('Prediction_views_ML/data/processed.csv')
df = df.loc[df['Visualizaciones']<35000,:] # Quito los 2 outliers

X_test = df.drop(columns=['Visualizaciones', 'Título del vídeo'])
y_test = df['Visualizaciones']

""" X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33, shuffle=True) """

with open('Prediction_views_ML/models/modelo_RFR.pkl', 'rb') as archivo_entrada:
    mejor_modelo = pickle.load(archivo_entrada)

""" mejor_modelo = RandomForestRegressor(
    criterion= 'absolute_error',
    max_depth=3,
    max_features=5,
    min_samples_leaf=2,
    min_samples_split=8) """

""" mejor_modelo.fit(X_train, y_train) """

y_pred = mejor_modelo.predict(X_test)

print("MAE:", mean_absolute_error(y_test, y_pred))
print("MSE:", mean_squared_error(y_test, y_pred))
print('RMSE:', np.sqrt(mean_squared_error(y_test, y_pred))) 