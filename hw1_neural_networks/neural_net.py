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

    total = x.shape[0]
    mask = list(range(total))
    np.random.shuffle(mask)
    x = x[mask]
    y = y[mask]
    train_split = int(0.8 * total)
    x_train, y_train = x[:train_split], y[:train_split]
    x_test, y_test = x[train_split:], y[train_split:]
    return x_train, y_train, x_test, y_test

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
    x_train, y_train, x_test, y_test = read_data()
    train_c0 = x_train[y_train == 0, :]
    train_c1 = x_train[y_train == 1, :]
    test_c0 = x_test[y_test == 0, :]
    test_c1 = x_test[y_test == 1, :]
    fig, a = plt.subplots(1, 2)
    fig.set_size_inches(11, 5)
    a[0].scatter(train_c0[:, 0], train_c0[:, 1], color='green', label='y = 0')
    a[0].scatter(train_c1[:, 0], train_c1[:, 1], color='red', label='y = 1')
    a[0].legend()
    a[0].set_title('Train Set')
    a[1].scatter(test_c0[:, 0], test_c0[:, 1], color='green', label='y = 0')
    a[1].scatter(test_c1[:, 0], test_c1[:, 1], color='red', label='y = 1')
    a[1].legend()
    a[1].set_title('Test Set')
    plt.show()

def q2_compute_gradient(x, y0, W, b):
    """
    This function computes the cost function with
    the given parameters.
    Then it computes the gradient of the cost with
    respect to W and b.
    """
    A = x @ W + b                                                   # A.shape = (180, )
    y = sigmoid(A)                                                  # y.shape = (180, )
    cost = -1 * np.sum(y0 * np.log(y) + (1 - y0) * np.log(1 - y))   # cost.shape = (180, )
    dy = -(y0 * (y ** -1) - (1 - y0) * ((1 - y) ** -1))            # dcost/dy
    dA = dy * (y * (1 - y))                       # dcost/dA
    dW = x.T @ dA                                                   # dcost/dW
    db = np.sum(dA)                                                 # dcost/db
    return cost, dW, db


def train_linear_network(x_train, y_train):
    W = np.random.normal(0, 1, (2, ))
    b = np.random.normal(0, 1, (1, ))
    n_epoch = 1000
    lr = 0.1
    for i in range(n_epoch):
        cost, dW, db = q2_compute_gradient(x_train, y_train, W, b)
        W -= lr * dW
        b -= lr * db
        print('epoch {}: cost = {}'.format(i+1, cost))
    return W, b

def q3_predict():
    x_train, y_train, x_test, y_test = read_data()
    W, b = train_linear_network(x_train, y_train)
    y_train_predicted = (x_train @ W + b >= 0.5) * 1
    y_test_predicted = (x_test @ W + b >= 0.5) * 1

    train_acc = np.sum(y_train_predicted == y_train) / x_train.shape[0]
    test_acc = np.sum(y_test_predicted == y_test) / x_test.shape[0]
    print('train accuracy =', train_acc)
    print('test accuracy =', test_acc)

q3_predict()
# q1_scatter_plot()
