import numpy as np

def computeCost(X, y, theta):
    m = len(y) # number of training examples
    J = np.sum((np.dot(X, theta) - y) ** 2) / (2 * m)
    return J
