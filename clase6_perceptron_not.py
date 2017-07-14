from sklearn.linear_model import perceptron
import numpy as np

# Function NOT
# NOT(1) = 0
# NOT(0) = 1

X = np.array([[1], [0]])
y = np.array([0, 1])

net = perceptron.Perceptron(n_iter=10, verbose=0)
net.fit(X, y)

print "Prediction:"
print "0 ->", net.predict(0)
print "1 ->", net.predict(1)
