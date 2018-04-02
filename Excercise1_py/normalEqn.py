import numpy as np

def normalEqn(X, y):
    X, y = np.matrix(X), np.matrix(y) # convert nparray into matrix
    theta = np.dot(np.dot(np.dot(X.T, X).I, X.T), y)
    return theta
