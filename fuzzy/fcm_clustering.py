"""
You must have the following installed:
    pandas, numpy, matplotlib, s-dbw
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import calinski_harabaz_score

def main():
    data = get_data('sample3.csv')
    # plot_data(data)
    # try_different_c(data, range(1, 12))
    C = 2
    m = 1.5
    centroids, U = cluster_data(data, C, m)
    plot_clusters(data, centroids, m)

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
    np.random.seed(35)
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
    costs, chs = [], []
    for c in c_set:
        centroids, U = cluster_data(data, c, m)
        cost = calculate_cost(data, centroids, U, m)
        ch = calculate_ch_index(data, centroids, U, m)
        costs.append(cost)
        chs.append(ch)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    fig.suptitle('Cost and CH index plots')
    ax1.plot(c_set, costs)
    ax1.set(xlabel='Number of clusters', ylabel='Cost')
    ax2.plot(c_set, chs)
    ax2.set(xlabel='Number of clusters', ylabel='Calinski_Harabaz index')
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

def calculate_ch_index(data, centroids, U, m):
    """
    According to each datum cluster label, this function
    calculates calinski_harabaz index.
    """
    labels = np.argmax(U, axis=1) + 1
    try:
        ch = calinski_harabaz_score(data, labels)
    except:
        ch = 0
    return ch

def plot_clusters(my_data, centroids, m):
    """
    Plots all data and centroids and colors them
    so we can see the cluster boundaries.
    """
    my_labels = np.argmax(calculate_U(my_data, centroids, m), axis=1) + 1
    all_data = np.random.rand(10000, 2)
    all_labels = np.argmax(calculate_U(all_data, centroids, m), axis=1) + 1
    plt.figure(figsize=(8,6))
    plt.scatter(all_data[:,0], all_data[:,1], s=0.1, c=all_labels)
    plt.scatter(my_data[:,0], my_data[:,1], s=20, c=my_labels)
    plt.scatter(centroids[:,0], centroids[:,1], s=40, c='g', marker='s')
    plt.show()

if __name__ == "__main__":
    main()
