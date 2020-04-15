import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def read_data():
    """
    This function reads the csv file and extracts data and labels
    as numpy array into variables names x and y and returns the result
    """
    csv_data = pd.read_csv('./dataset.csv')
    x = csv_data[['X1', 'X2']]
    x = x.values                                # numpy array for x: (180, 2)
    y = csv_data['Label']
    y = y.values                                # numpy array for y: (180, )
    return x, y

def sigmoid(x):
    """
    This function computes the sigmoid function for input of any shape.
    """
    s = 1 / (1 + np.exp(-x))
    return s

def q1_scatter_plot():
    """
    This function plots the data as a scatter plot using
    matplotlib library.
    """
    x, y = read_data()
    class0 = x[y == 0, :]
    class1 = x[y == 1, :]
    plt.scatter(class0[:, 0], class0[:, 1], color='green', label='y = 0')
    plt.scatter(class1[:, 0], class1[:, 1], color='red', label='y = 1')
    plt.legend()
    plt.show()

def q2_compute_gradient(x, y0, W, b):
    """
    This function computes the cost function with
    the given parameters.
    Then it computes the gradient of the cost with
    respect to W and b.
    """
    A = x @ W + b
    assert A.shape == (180, ), "A has wrong shape"
    y = sigmoid(A)
    # TODO: fix cost function
    cost = -np.sum(y0 * np.log(y) + (1 - y0) * np.log(1 - y))

    dy = - (y0 * (y ** -1) + (1 - y0) * ((1 - y) ** -1))
    dA = dy * (sigmoid(A) * (1 - sigmoid(A)))
    assert dA.shape == (180, ), "dA has wrong shape"
    dW = x.T @ dA
    db = np.sum(dA)
    return cost, dW, db

