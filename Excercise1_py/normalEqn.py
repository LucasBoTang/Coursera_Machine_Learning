import numpy as np

def normalEqn(X, y):
    theta = np.dot(np.dot(np.dot(X.T, X).I, X.T), y)
    return theta
