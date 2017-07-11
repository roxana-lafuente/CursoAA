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

X = data_train[[u'sepal length (cm)', u'sepal width (cm)']]
y = data_train[u'target']

# Crear la figura
title = "Regularization vs No regularization"
fig = plt.figure(title)
ax = fig.add_subplot(1, 2, 1)
ax.set_title("No regularization")

h = .02  # Separacion entre los valores a generar

# No uso regularization
model = LogisticRegression(multi_class='ovr')

# Entrenamos nuestro clasificador con len(X) observaciones, cada una con su target y
model.fit(X, y)

x_min, x_max = X[u'sepal length (cm)'].min() - 0.05, X[u'sepal length (cm)'].max() + 0.05
y_min, y_max = X[u'sepal width (cm)'].min() - 0.05, X[u'sepal width (cm)'].max() + 0.05

# Generar valores entre N1 y N2 con separacion de h usando np.arange.
# Luego usar meshgrid:
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

# Plot
plt.scatter(X[u'sepal length (cm)'], X[u'sepal width (cm)'],
            c=y, cmap=plt.cm.Oranges)

# Nombrar los ejes para entender mejor la figura.
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')



# Real values
ax = fig.add_subplot(1, 2, 2)
ax.set_title("Regularization")

# TODO: Agregar el codigo para predecir con regularization.
# Probar con diferentes C.

# Nombrar los ejes para entender mejor la figura.
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.show()
