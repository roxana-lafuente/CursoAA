from sklearn.datasets import load_boston
from statsmodels.api import add_constant, OLS
from sklearn.cross_validation import train_test_split
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
DATA SET INFO:
Feature names:
    'CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT'
Attribute Information (in order):
    - CRIM     per capita crime rate by town
    - ZN       proportion of residential land zoned for lots over 25,000 sq.ft.
    - INDUS    proportion of non-retail business acres per town
    - CHAS     Charles River dummy variable (= 1 if tract bounds river; 0 otherwise)
    - NOX      nitric oxides concentration (parts per 10 million)
    - RM       average number of rooms per dwelling
    - AGE      proportion of owner-occupied units built prior to 1940
    - DIS      weighted distances to five Boston employment centres
    - RAD      index of accessibility to radial highways
    - TAX      full-value property-tax rate per $10,000
    - PTRATIO  pupil-teacher ratio by town
    - B        1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
    - LSTAT    Percentage lower status of the population
    - MEDV     Median value of owner-occupied homes in $1000's    
Number of Instances:
    506
"""

# Cargar conjunto de datos
dataset = load_boston()
boston = pd.DataFrame(dataset.data, columns=dataset.feature_names)
boston['MEDV'] = dataset.target

# Dividir en conjunto de testing y de training.
boston_train, boston_test = train_test_split(boston, test_size=0.3)

# Ingenieria de caracteristicas: seleccion
#print boston.corr(method='pearson')

# Seleccionar que atributos usar en la regresion
X = boston_train['LSTAT']
y = np.log(boston_train['MEDV'])

# Regresion
model = OLS(y, add_constant(X))
model = model.fit()
theta = model.params

# Plot toda la info
X = boston['LSTAT']
y = np.log(boston['MEDV'])

print "Estimated parameters:\n", theta

# Prepare plots.
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot.
ax.scatter(X, y, label='Dataset', color='Cyan')

# Mostrar la regresion lineal.
x = np.linspace(X.min(), X.max(), len(X))
ax.plot(x, model.predict(add_constant(x)), 'r', label='OLS', color='Green')

# Plot settings.
ax.set_xlabel('log(LSTAT)')
ax.set_ylabel('log(MEDV)')
ax.set_title("log(MEDV) vs log(LSTAT)")
ax.legend()



# Prepare plots.
fig1, ax1 = plt.subplots(figsize=(12, 8))

# Scatter plot.
ax1.scatter(X, np.exp(y), label='Dataset', color='Cyan')

# Mostrar la regresion lineal.
x = np.linspace(X.min(), X.max(), len(X))
ax1.plot(x, np.exp(model.predict(add_constant(x))), 'r', label='OLS', color='Green')

# Plot settings.
ax1.set_xlabel('LSTAT')
ax1.set_ylabel('MEDV')
ax1.set_title("MEDV vs LSTAT")
ax1.legend()

plt.show()
