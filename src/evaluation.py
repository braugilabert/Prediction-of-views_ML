import pandas as pd
import numpy as np

test = pd.read_csv('ML Project/data/test.csv') 

X = test[['Med Juego Compartido']]
y = test[['Visualizaciones']]

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=33)

lr = LinearRegression()
lr.fit(X_train,y_train)

predicciones_finales = lr.predict(X_test)

from sklearn import metrics
print("MAE", metrics.mean_absolute_error(y_test, predicciones_finales))
print("MSE", metrics.mean_squared_error(y_test, predicciones_finales))
print("RMSE", np.sqrt(metrics.mean_squared_error(y_test, predicciones_finales)))
# print("MAPE", mean_absolute_percentage_error(diabetes_target_test, predictions)) #probar