import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV #Hecho en notebooks. Los parámetros que salen aqui son los mejores.
from sklearn.model_selection import cross_val_score #no lo he usado
from xgboost import XGBRegressor
import pickle
import yaml

df = pd.read_csv('Prediction_views_ML/data/processed.csv')
df = df.loc[df['Visualizaciones']<35000,:] #quito 2 outliers

X = df.drop(columns=['Visualizaciones', 'Título del vídeo'])
y = df['Visualizaciones']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33, shuffle=True)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

regressor = XGBRegressor()
model = regressor.fit(X_train, y_train)
y_pred = model.predict(X_test)

# XGBRegressor con la mejor hiperparametrización (tras haber hecho GridSearchCV de 50 minutos)

regressor = XGBRegressor(
    gamma=0.05,
    learning_rate=0.1,
    max_depth=5,
    n_estimators=100,
    objective='reg:squarederror',
    subsample=0.2,
    scale_pos_weight=0,
    reg_alpha=0,
    reg_lambda=1
)
model = regressor.fit(X_train, y_train)
y_pred = model.predict(X_test)

import pickle
with open('Prediction_views_ML/models/modelo_XGB.pkl', 'wb') as archivo:
    pickle.dump(model, archivo)
    
with open('Prediction_views_ML/models/modelo_config_def.yaml', 'w') as c:
    yaml.dump(model, c)