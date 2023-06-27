import pandas as pd
import numpy as np
import pickle
from sklearn import metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_squared_error, mean_absolute_percentage_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

with open('Prediction_views_ML/models/modelo_XGB.pkl', 'rb') as f:
    model = pickle.load(f)

test = pd.read_csv('Prediction_views_ML/data/test.csv')
test = test.loc[test['Visualizaciones']<35000,:] #quito 2 outliers para ver que tal

X = test.drop(columns=['Visualizaciones', 'Título del vídeo'])
y = test['Visualizaciones']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=33, shuffle=True)

predicciones = model.predict(X_test)
predicciones

predicciones = model.predict(X_test)

model.score(X_test, y_test) #revisar

mse = mean_squared_error(y_test,predicciones) #REVISAR
mae = metrics.mean_absolute_error(y_test, predicciones)
rmse = np.sqrt(metrics.mean_squared_error(y_test, predicciones))
mape = mean_absolute_percentage_error(y_test, predicciones)

print("Mean Squared Error:", mse)
print("Mean Absolute Error:", mse)
print("RMSE:", rmse)
print("MAPE", mape)