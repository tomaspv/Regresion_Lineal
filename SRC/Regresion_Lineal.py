# -*- coding: utf-8 -*-
"""
Modelo de regresion lineal en Python

@author: TPV
"""


# Importamos librerias con las que vamos a trabajar

#Para trabajar con matrices y realizar operaciones matematicas
import numpy as np
#Para hacer los graficos
import matplotlib.pyplot as plt
#Para trabajar con los datos
import pandas as pd

#Importamos el data set
ds = pd.read_excel('../DATASET/Wine.xls')

#Separamos las variables para trabajarlas
x = ds.loc[:, ['Alcohol']].values
y = ds.loc[:, ['Color intensity']].values

#Observamos si a priori existe una correlacion entre las variables
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

#intercept - ordenada al origen
regresor.intercept_

#coef - Por cada unidad de alcohol que aumenta, se incrementa en 1,69 el coeficiente de color del vino
regresor.coef_


#####################
#Evaluamos el modelo#
#####################

#Utilizamos la libreria statsmodels para hallar el valor de Tstudent y Pvalue
import statsmodels.api as sm

#Analizamos Tstudent y Pvalue
#Si el valor de Tstudent es cero, diremos que no hay relacion entre las variables X e Y.
#Si el valor de P-Value es muy bajo se suelen tomar valores menores a 0,05 para considerar que es bajo, decimos que hay relacion entre las variables X e Y

est = sm.OLS(y, x)
est2 = est.fit()
print(est2.summary())

#Creamos una tabla para observar el valor real del numero y la prediccion que realizo el modelo.
pred_ds = pd.DataFrame(y_pred, columns=['Prediccion'])
pred_ds = pred_ds.assign(Valor_Real= y_test)
pred_ds

#Utilizamos la libreria sklearn para halar metricas
from sklearn import metrics

#EVS - Explanied_variance_score
#Es una medida que me indica que tan dispersos son los datos que estoy analizando. Oscila entre 0 y 1. Mientas mas cerca este de 1 mejor.
metrics.explained_variance_score(y_test,y_pred)

#MAE - Mean Obsolute error
#Es la media absoluta de los errores
metrics.mean_absolute_error(y_test, y_pred)

#MSE - Mean Squared error
#Es el MAE elevado al cuadrado, castiga a los errores que son mas grandes.
metrics.mean_squared_error(y_test, y_pred)

#RMSE - Root Mean Squared Error
#Es el MSE pero la raiz, es interpretable en la unidad de la variable dependiente "Y"
np.sqrt(metrics.mean_squared_error(y_test, y_pred))

#R^2 (Multiple R-Squared) es un indicador de bondad de ajuste de mi modelo.R^2 oscila entre 0 y 1. Mientras
#mas se acerque a 1 , indica que el modelo se ajusta bien a los datos.
metrics.r2_score(y_test,y_pred)


#####################
#Graficamos el modelo#
#####################

# Ploteamos el conjunto de entrenamiento
plt.scatter(x_train, y_train, color = 'red')
plt.plot(x_train, regresor.predict(x_train), color = 'blue')
plt.title('Alcohol vs Intensidad del color (Training set)')
plt.xlabel('Alcohol')
plt.ylabel('Intensidad del color')
plt.show()

# Ploteamos el conjunto de prueba
plt.scatter(x_test, y_test, color = 'red')
plt.plot(x_train, regresor.predict(x_train), color = 'blue')
plt.title('Alcohol vs Intensidad del color (Test set)')
plt.xlabel('Alcohol')
plt.ylabel('Intensidad del color')
plt.show()





