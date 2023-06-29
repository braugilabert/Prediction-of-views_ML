import yaml
import pickle
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

df = pd.read_csv('Prediction_views_ML/data/test.csv')
df = df.loc[df['Visualizaciones']<35000,:] # Quito los 2 outliers

X = df.drop(columns=['Visualizaciones', 'Título del vídeo'])
y = df['Visualizaciones']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33, shuffle=True)

mejor_modelo = RandomForestRegressor(
    criterion= 'absolute_error',
    max_depth=3,
    max_features=5,
    min_samples_leaf=2,
    min_samples_split=8)

mejor_modelo.fit(X_train, y_train)

y_pred = mejor_modelo.predict(X_test)

import pickle
with open('Prediction_views_ML/models/modelo_RFR.pkl', 'wb') as archivo:
    pickle.dump(mejor_modelo, archivo)
    
with open('Prediction_views_ML/models/modelo_config_def.yaml', 'w') as c:
    yaml.dump(mejor_modelo, c)