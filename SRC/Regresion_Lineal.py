# Importamos librerias con las que vamos a trabajar

#Para 
import numpy as np
#Para hacer los graficos
import matplotlib.pyplot as plt
#Para trabajar con los datos
import pandas as pd

#Importamos el data set
ds = pd.read_csv('../DATASET/student/student-mat.csv',sep =';')
ds = pd.read_excel('../DATASET/Wine.xls')

#Separamos las variables para trabajarlas
x = ds.loc[:, ['Alcohol']].values
y = ds.loc[:, ['Color intensity']].values

plt.scatter(x,y)

#Dividimos el data set en dos partes. Entrenamiento y prueba.
from sklearn.cross_validation import train_test_split
x_train, x_test,y_train, y_test = train_test_split(x,y,test_size = 1/3, random_state = 0)

#Ajustamos el modelo de entrenamiento
from sklearn.linear_model import LinearRegression
regresor = LinearRegression()
regresor.fit(x_train, y_train)

#Ejecutamos el modelo
y_pred = regresor.predict(x_test)

#Evaluamos el modelo#
#####################
#R^2 (Multiple R-Squared) es un indicador de bondad de ajuste de mi modelo.R^2 oscila entre 0 y 1. Mientras
#mas se acerque a 1 , indica que el modelo se ajusta bien a los datos.
regresor.score(x_test,y_pred)

metrics

#intercept - ordenada al origen
regresor.intercept_

#coef - Por cada unidad de alcohol que aumenta, se incrementa en 1,69 el coeficiente de color del vino
regresor.coef_

#MAE - Mean Obsolute error
#Es la media absoluta de los errores
from sklearn import metrics
metrics.mean_absolute_error(y_test, y_pred)

#MSE - Mean Squared error
#Es el MAE elevado al cuadrado, castiga a los errores que son mas grandes.
metrics.mean_squared_error(y_test, y_pred)

#RMSE - Root Mean Squared Error
#Es el MSE pero la raiz, es interpretable en la unidad de la variable dependiente "Y"
np.sqrt(metrics.mean_squared_error(y_test, y_pred))



# Visualising the Training set results
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regresor.predict(x_train), color = 'blue')
plt.title('Alcohol vs Intensidad del color (Training set)')
plt.xlabel('Alcohol')
plt.ylabel('Intensidad del color')
plt.show()

# Visualising the Test set results
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regresor.predict(x_train), color = 'blue')
plt.title('Alcohol vs Intensidad del color (Test set)')
plt.xlabel('Alcohol')
plt.ylabel('Intensidad del color')
plt.show()