import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    data = get_data('sample1.csv')
    plot_data(data)
    C = find_optimum_c(data)
    centroids, cost, pbmf = cluster_data(data, C)
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

def cluster_data(data, C):
    pass

def plot_clusters(data, centroids):
    pass

if __name__ == "__main__":
    main()