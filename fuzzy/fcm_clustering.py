import pandas as pd
import numpy as np

def main():
    data = get_data('sample1.csv')
    plot_data(data)
    C = find_optimum_c(data)
    centroids, cost, pbmf = cluster_data(data, C)
    plot_clusters(data, centroids)

def get_data(file):
    pass

def plot_data(data):
    pass

def find_optimum_c(data):
    pass

def cluster_data(data, C):
    pass

def plot_clusters(data, centroids):
    pass

if __name__ == "__main__":
    main()