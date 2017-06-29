from sklearn.datasets import load_boston
from statsmodels.api import add_constant, OLS
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

# Ingenieria de caracteristicas: seleccion
print boston.corr(method='pearson')

# Seleccionar que atributos usar en la regresion
X = add_constant(boston['LSTAT'])
y = boston['MEDV']

# Regresion
model = OLS(y, X)
model = model.fit()
theta = model.params

print "Theta:\n", theta

# Prepare plots.
fig, ax = plt.subplots(figsize=(12, 8))

# Scatter plot.
ax.scatter(boston['MEDV'], boston['LSTAT'], label='Dataset', color='Cyan')


# Plot settings.
ax.set_xlabel('LSTAT')
ax.set_ylabel('MEDV')
ax.set_title("MEDV vs LSTAT")
ax.legend()
plt.show()
