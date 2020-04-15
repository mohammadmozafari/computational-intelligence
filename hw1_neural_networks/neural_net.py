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


def q1_scatter_plot(x_train, y_train, x_test, y_test):
    """
    This function plots the data as a scatter plot using
    matplotlib library.
    """
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

def q2_compute_cost_gradient(x, y0, W, b):
    """
    This function computes the cost function with
    the given parameters.
    Then it computes the gradient of the cost with
    respect to W and b.
    """
    A = x @ W + b                                                   # A.shape = (180, )
    y = sigmoid(A)                                                  # y.shape = (180, )
    if y0 is None:
        return y
    
    cost = -1 * np.sum(y0 * np.log(y) + (1 - y0) * np.log(1 - y))   # cost.shape = (180, )
    dy = -(y0 * (y ** -1) - (1 - y0) * ((1 - y) ** -1))             # dcost/dy
    dA = dy * (y * (1 - y))                                         # dcost/dA
    dW = x.T @ dA                                                   # dcost/dW
    db = np.sum(dA)                                                 # dcost/db
    return cost, dW, db

def q5_compute_cost_gradient(x, y0, W, V, U, b0, b1, b2):
    """
    This function computes the cost function with
    the given parameters for the second neural network.
    Then it computes the gradient of the cost with
    respect to parameters: W, V, U, b0, b1, b2
    """
    A1 = x @ W + b0
    A2 = x @ V + b1
    z0 = sigmoid(A1)
    z1 = sigmoid(A2)
    z = np.array([z0, z1]).T
    A3 = z @ U + b2
    y = sigmoid(A3)
    if y0 is None:
        return y

    cost = np.sum((y - y0) ** 2)
    dy = 2 * (y - y0)                                               # (180, )
    dA3 = dy * (y * (1 - y))                                        # (180, )
    dz0 = dA3 * U[0]
    dz1 = dA3 * U[1]
    dA1 = dz0 * (z0 * (1 - z0))
    dA2 = dz1 * (z1 * (1 - z1))
    dW = x.T @ dA1
    dV = x.T @ dA2
    dU = z.T @ dA3
    db0 = np.sum(dA1)
    db1 = np.sum(dA2)
    db2 = np.sum(dA3)
    return cost, dW, dV, dU, db0, db1, db2

def q3_train_network(x_train, y_train):
    """
    This function initializes W and b using normal distribution
    and then uses train data to train them.
    It returns W and b.
    """
    W = np.random.normal(0, 1, (2, ))
    b = np.random.normal(0, 1, (1, ))
    n_epoch = 10000
    lr = 0.01
    for i in range(n_epoch):
        cost, dW, db = q2_compute_cost_gradient(x_train, y_train, W, b)
        W -= lr * dW
        b -= lr * db
        print('epoch {}: cost = {}'.format(i+1, cost))
    return W, b

def q5_train_network(x_train, y_train):
    """
    This function initializes W, V, U and b0, b1, b2 using normal distribution
    and then uses train data to train them.
    It returns W, V, U and b0, b1, b2.
    """
    W = np.random.normal(0, 1, (2, ))
    V = np.random.normal(0, 1, (2, ))
    U = np.random.normal(0, 1, (2, ))
    b0 = np.random.normal(0, 1, (1, ))
    b1 = np.random.normal(0, 1, (1, ))
    b2 = np.random.normal(0, 1, (1, ))
    n_epoch = 1000
    lr = 0.3
    for i in range(n_epoch):
        cost, dW, dV, dU, db0, db1, db2 = q5_compute_cost_gradient(x_train, y_train, W, V, U, b0, b1, b2)
        W -= (lr * dW)
        V -= (lr * dV)
        U -= (lr * dU)
        b0 -= (lr * db0)
        b1 -= (lr * db1)
        b2 -= (lr * db2)
        print('epoch {}: cost = {}'.format(i+1, cost))
    return W, V, U, b0, b1, b2


def q4_q5_predict(x_train, y_train, x_test, y_test, fn, params):
    """
    This function takes the train weights for the linear model
    and predicts train and test labels. Then it plots the result.
    """
    y_train_predicted = fn(x_train, None, *params)
    y_train_predicted = (y_train_predicted >= 0.5) * 1
    y_test_predicted = fn(x_test, None, *params)
    y_test_predicted = (y_test_predicted >= 0.5) * 1

    train_acc = np.sum(y_train_predicted == y_train) / x_train.shape[0]
    test_acc = np.sum(y_test_predicted == y_test) / x_test.shape[0]
    print('train accuracy =', train_acc)
    print('test accuracy =', test_acc)
    q1_scatter_plot(x_train, y_train_predicted, x_test, y_test_predicted)

if __name__ == '__main__':
    x_train, y_train, x_test, y_test = read_data()
    # q1_scatter_plot(x_train, y_train, x_test, y_test)
    params = q5_train_network(x_train, y_train)
    # params = q5_train_network(x_train, y_train)
    q4_q5_predict(x_train, y_train, x_test, y_test, q5_compute_cost_gradient, params)
