from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.cross_validation import train_test_split
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

data_train, data_test = train_test_split(data, test_size=0.3)

# Scatter plot.
# Notar que dataset.data es un numpy.ndarray con shape (150, 4)
X = data_train[[u'sepal length (cm)', u'sepal width (cm)']]
y = data_train[u'target']

X0 = data_train[[u'sepal length (cm)', u'sepal width (cm)', u'petal length (cm)', u'petal width (cm)']]
y0 = data_train[u'target']

model0 = LogisticRegression(C=1e5, multi_class='ovr')
model0.fit(X0, y0)

# Crear figura.
title = "Sepal Scatter Plot"
plt.figure(title, figsize=(8, 6))

# Clasificacion - Creamos una instancia del clasificador Logistic Regression
model = LogisticRegression(C=1e5, multi_class='ovr')

# Entrenamos nuestro clasificador con len(X) observaciones, cada una con su target y
model.fit(X, y)

h = .02  # Separacion entre los valores a generar

# Para poder mostrar el limite de decision, generamos puntos en el rango
# [x_min - 0.05, x_max + 0.05] * [y_min - 0.05, y_max + 0.05].
# Obtener minimo y maximo de cada variable.
x_min, x_max = X[u'sepal length (cm)'].min() - 0.05, X[u'sepal length (cm)'].max() + 0.05
y_min, y_max = X[u'sepal width (cm)'].min() - 0.05, X[u'sepal width (cm)'].max() + 0.05

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
plt.scatter(X[u'sepal length (cm)'], X[u'sepal width (cm)'], c=y, cmap=plt.cm.Oranges)

# Nombrar los ejes para entender mejor la figura.
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# Mostrar figura.
plt.show()


# Print metrics
print "Accuracy"
X_test0 = np.c_[data_test[u'sepal length (cm)'].ravel(),
                data_test[u'sepal width (cm)'].ravel(),
                data_test[u'petal length (cm)'].ravel(),
                data_test['petal width (cm)'].ravel()]
X_test = np.c_[data_test[u'sepal length (cm)'].ravel(),
               data_test[u'sepal width (cm)'].ravel()]
print "Model0", accuracy_score(data_test[u'target'], model0.predict(X_test0))
print "Model", accuracy_score(data_test[u'target'], model.predict(X_test))

print "\n\nPrecision"
print "Model0", precision_score(data_test[u'target'], model0.predict(X_test0), average='macro')
print "Model", precision_score(data_test[u'target'], model.predict(X_test), average='macro')

print "\n\nRecall"
print "Model0", recall_score(data_test[u'target'], model0.predict(X_test0), average='macro')
print "Model", recall_score(data_test[u'target'], model.predict(X_test), average='macro')
