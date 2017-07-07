from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Cargar conjunto de datos.
dataset = load_iris()


# Transformar a pandas dataframe para ver un resumen de los datos.
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
X = data[[u'sepal length (cm)', u'sepal width (cm)']]
y = dataset['target']

# Crear la figura
title = "Clustering with K-Means vs Real"
fig = plt.figure(title)
ax = fig.add_subplot(1, 2, 1)
ax.set_title("Clustering")

# Elegir el numero de clusters
model = KMeans(n_clusters=1)
model.fit(X)
# Obtener para cada X el cluster que tiene asignado
labels = y

# Plot
plt.scatter(X[u'sepal length (cm)'], X[u'sepal width (cm)'],
            c=labels.astype(np.float), cmap=plt.cm.Oranges)

# Nombrar los ejes para entender mejor la figura.
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')



# Real values
ax = fig.add_subplot(1, 2, 2)
ax.set_title("Real values")

# TODO: Agregar el codigo para mostrar los valores reales.

# Nombrar los ejes para entender mejor la figura.
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

plt.show()
