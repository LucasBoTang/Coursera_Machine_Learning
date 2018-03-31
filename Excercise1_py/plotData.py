from matplotlib import pyplot as plt

def plotData(X, y):
    plt.scatter(X, y, color='red', marker='x', s=10)
    plt.xlabel('Population')
    plt.ylabel('Revenue')
    plt.show()
