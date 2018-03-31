import numpy as np
from matplotlib import pyplot as plt

## ==================== Part 1: Basic Function ====================
print('Running warmUpExercise ... \n')
print('5x5 Identity Matrix: \n')

from warmUpExercise import warmUpExercise
warmUpExercise()

print('\nProgram paused. Press enter to continue.\n')
input()


## ======================= Part 2: Plotting =======================
print('Plotting Data ...\n')
data = np.genfromtxt('ex1data1.txt', delimiter = ',')
X = data[:, 0]
y = data[:, 1]
m = len(y) # number of training example

from plotData import plotData
plotData(X, y)

print('\nProgram paused. Press enter to continue.\n')
input()


## =================== Part 3: Cost and Gradient descent ===================
X = np.c_[np.ones((m, 1)), X] # add a column of ones to x
y = np.array([y]).T # make sure y is a column vector
theta = np.zeros((2, 1)) # initialize fitting parameters

# some gradient descent settings
iterations = 1500
alpha = 0.01

print('\nTesting the cost function ...\n')

# compute and display initial cost
from computeCost import computeCost
J = computeCost(X, y, theta)
print('With theta = (0, 0)\nCost computed = {:.2f}\n'.format(J))
print('Expected cost value (approx) 32.07\n')

# further testing of the cost function
J = computeCost(X, y, [[-1], [2]]);
print('\nWith theta = (-1, 2)\nCost computed {:.2f}\n'.format(J))
print('Expected cost value (approx) 54.24\n')

print('\nProgram paused. Press enter to continue.\n')
input()

print('\nRunning Gradient Descent ...\n')
# run gradient descent
from gradientDescent import gradientDescent
theta = gradientDescent(X, y, theta, alpha, iterations)

# print theta to screen
print('Theta found by gradient descent:\n')
print('{:.4f}\n{:.4f}\n'.format(theta[0][0], theta[1][0]))
print('Expected theta values (approx)\n')
print(' -3.6303\n  1.1664\n\n')

print('\nProgram paused. Press enter to continue.\n')
input()

# plot the linear fit
plt.scatter(X[:, 1], y, color='red', marker='x', s=10)
plt.plot(X[:, 1], np.dot(X, theta), '-')
plt.xlabel('Population')
plt.ylabel('Revenue')
plt.legend(['Training data', 'Linear regression'])
plt.show()

print('\nProgram paused. Press enter to continue.\n')
input()

## ============= Part 4: Visualizing J(theta_0, theta_1) =============
print('Visualizing J(theta_0, theta_1) ...\n')

# grid over which we will calculate J
theta0_vals = np.linspace(-10, 10, 100)
theta1_vals = np.linspace(-1, 4, 100)

# initialize J_vals to a matrix of 0's
J_vals = np.zeros((len(theta0_vals), len(theta1_vals)))

# fill out J_vals
for i in range(len(theta0_vals)):
    for j in range(len(theta1_vals)):
        t = [[theta0_vals[i]], [theta1_vals[j]]]
        J_vals[i, j] = computeCost(X, y, t)

# because of the way meshgrids work in the surf command, we need to
# transpose J_vals before calling surf, or else the axes will be flipped
J_vals = J_vals.T

# contour plot
# plot J_vals as 15 contours spaced logarithmically between 0.01 and 100
plt.contour(theta0_vals, theta1_vals, J_vals, np.logspace(-2, 3, 20))
plt.xlabel('theta_0')
plt.ylabel('theta_1')
plt.scatter(theta[0][0], theta[1][0], color='red', marker='x', s=10)
plt.show()
