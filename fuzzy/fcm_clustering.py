import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = get_data('sample1.csv')
    # plot_data(data)
    # C = find_optimum_c(data)
    C = 4
    m = 1.5
    centroids, cost, pbmf = cluster_data(data, C, m)
    plot_clusters(data, centroids)

def get_data(file):
    """
    Extracts a 2d numpy array out of
    the given csv file. each row consists of
    2 numbers, the coordiates of the datum.
    """
    data = pd.read_csv(file)
    data = data.values
    print('Shape of data:', data.shape)
    return data

def plot_data(data):
    """
    Plots the data extracted from csv file
    in a 2d plane.
    """
    plt.scatter(data[:,0], data[:, 1], s=12, c='g')
    plt.xlabel('X 1')
    plt.ylabel('X 2')
    plt.show()

def find_optimum_c(data):
    pass

def cluster_data(data, C, m=1.5, eps=1e-5):
    """
    Executes the FCM algorithm until convergence.
    C is the number of centroids and m is the fuzzifier.
    """
    U = np.random.rand(data.shape[0], C)
    U = (U.T/U.sum(axis=1)).T
    while True:
        um = U ** m
        c = um.T @ data
        c = (c.T/um.sum(axis=0)).T
        oldU = U.copy()
        U = -2 * data @ c.T
        U += np.sum(c**2, axis=1)
        U = (U.T + np.sum(data**2, axis=1)).T
        U = U ** (1/(m-1))
        U = 1/U
        U = (U.T/U.sum(axis=1)).T
        error = np.sum(np.abs(U-oldU))
        print('diff:', error)
        if error < eps:
            break

def plot_clusters(data, centroids):
    pass

if __name__ == "__main__":
    main()