from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


# Cargar conjunto de datos.
dataset = load_iris()


# Transformar a pandas dataframe para ver un resumen de los datos.
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
data['target'] = dataset['target']

# Scatter plot.
# Notar que dataset.data es un numpy.ndarray con shape (150, 4)
X = dataset.data[:, :2]  # solo queremos las primeras dos columnas.
y = dataset.target

# Crear figura.
title = "Sepal Scatter Plot"
plt.figure(title, figsize=(8, 6))

# Clasificacion - Creamos una instancia del clasificador Logistic Regression
model = LogisticRegression(C=1e5)

# Entrenamos nuestro clasificador con len(X) observaciones, cada una con su target y
model.fit(X, y)

h = .02  # Separacion entre los valores a generar

# Para poder mostrar el limite de decision, generamos puntos en el rango
# [x_min - 0.05, x_max + 0.05] * [y_min - 0.05, y_max + 0.05].
# Obtener minimo y maximo de cada variable.
x_min, x_max = X[:, 0].min() - 0.05, X[:, 0].max() + 0.05
y_min, y_max = X[:, 1].min() - 0.05, X[:, 1].max() + 0.05

# Generar valores entre N1 y N2 con separacion de h usando np.arange.
# Luego usar meshgrid:
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Para nuestros puntos, predecir a que clase pertenece cada uno.
# ravel() transforma matrices en vectores
Z = model.predict(np.c_[xx.ravel(), yy.ravel()])

# Transformamos de vector a matriz
Z = Z.reshape(xx.shape)
# Ahora mostramos los limites de decision
plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Crear scatter plot sobre los limites de decision.
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Oranges)

# Nombrar los ejes para entender mejor la figura.
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# Mostrar figura.
plt.show()
