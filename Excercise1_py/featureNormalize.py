import numpy as np

def featureNormalize(X):
    mu = np.mean(X, axis=0)
    sigma = np.std(X, axis=0)
    print(mu, sigma)
    return X, mu, sigma
