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

q1_scatter_plot()
