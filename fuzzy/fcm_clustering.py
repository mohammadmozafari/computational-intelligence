import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = get_data('sample4.csv')
    # plot_data(data)
    try_different_c(data, range(1, 10))
    # C = 4
    # m = 1.5
    # centroids = cluster_data(data, C, m)
    # plot_clusters(data, centroids)

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

def cluster_data(data, C, m=1.5, eps=1e-5):
    """
    Executes the FCM algorithm until convergence.
    C is the number of centroids and m is the fuzzifier.
    """
    U = np.random.rand(data.shape[0], C)
    U = (U.T/U.sum(axis=1)).T
    while True:
        um = U ** m
        centroids = um.T @ data
        centroids = (centroids.T/um.sum(axis=0)).T
        oldU = U.copy()
        U = calculate_U(data, centroids, m)
        error = np.sum(np.abs(U-oldU))
        print('diff:', error)
        if error < eps:
            break
    return centroids, U

def calculate_U(data, centroids, m):
    """
    Uses the distribution of data and current
    cluster centroids to determine how much each
    datum belongs to each cluster.
    """
    U = -2 * data @ centroids.T
    U += np.sum(centroids**2, axis=1)
    U = (U.T + np.sum(data**2, axis=1)).T
    U = U ** (1/(m-1))
    U = 1/U
    U = (U.T/U.sum(axis=1)).T
    return U

def try_different_c(data, c_set, m=1.5):
    """
    Clusters data with different number of clusters
    and returns results to choose the best c.
    """
    costs = []
    for c in c_set:
        centroids, U = cluster_data(data, c, m)
        cost = calculate_cost(data, centroids, U, m)
        costs.append(cost)
    plt.plot(c_set, costs)
    plt.show()

def calculate_cost(data, centroids, U, m):
    """
    Using the distribution of data, cluster centroids
    and the membership of each datum to each cluster
    calculates the cost of clustering.
    """
    distance = -2 * data @ centroids.T
    distance += np.sum(centroids**2, axis=1)
    distance = (distance.T + np.sum(data**2, axis=1)).T
    cost = np.sum(distance * (U ** m))
    return cost

def plot_clusters(data, centroids):
    pass

if __name__ == "__main__":
    main()
