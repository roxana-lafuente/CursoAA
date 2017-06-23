from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
import pandas as pd


# Cargar conjunto de datos.
dataset = load_iris()


# Transformar a pandas dataframe para ver un resumen de los datos.
data = pd.DataFrame(dataset.data, columns=dataset.feature_names)
data['target'] = dataset['target']

# Analisis descriptivo.
print data.describe()


# Scatter plot.
# Notar que dataset.data es un numpy.ndarray con shape (150, 4)
X = dataset.data[:, :2]  # solo queremos las primeras dos columnas.
Y = dataset.target

# Crear figura.
title = "Sepal Scatter Plot"
plt.figure(title, figsize=(8, 6))

# Crear scatter plot.
plt.scatter(X[:, 0], X[:, 1], c=Y, cmap=plt.cm.Paired)

# Nombrar los ejes para entender mejor la figura.
plt.xlabel('Sepal length')
plt.ylabel('Sepal width')

# Mostrar figura.
plt.show()
