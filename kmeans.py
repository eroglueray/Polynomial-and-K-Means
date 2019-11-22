# -*- coding: utf-8 -*-
"""
@author: Joe Pearson 14587506
References:
Description To Understand with Sklearn, before attempting without: 
    https://towardsdatascience.com/understanding-k-means-clustering-in-machine-learning-6a6e67336aa1
Euclidean Distance:
    https://stackoverflow.com/questions/1401712/how-can-the-euclidean-distance-be-calculated-with-numpy
"""
# Import data from file before extracting the right values
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

X =pd.read_csv('#')

# Stack the height and tail_length arrays (column data from X) for use in the code below
height = X['height'].values
tail_length = X['tail length'].values
leg_length = X['leg length'].values

height.shape
tail_length.shape
leg_length.shape

datapoints = np.vstack((height, tail_length)).T
datapoints2 = np.vstack((height, leg_length)).T

# Functions for K-Means
def compute_euclidean_distance(vec_1, vec_2):
    # Determines the distance using euclidean distance formula
    distance = np.linalg.norm(vec_2 - vec_1, axis=1)
    return distance

def initialise_centroids(dataset, k):
    # Initialise centroid based on random int within the shape of the dataset given
    # to the size of the k supplied.
    
   #centroids = dataset.copy()
    centroids = dataset[np.random.randint(dataset.shape[0], size=k)]
    return centroids[:k]

# Kmeans function to return the clusters and their relevant centroid
def kmeans(dataset, k):
    # Initialize the centroids for use
    centroids = initialise_centroids(dataset, k)
    
    # Initialize the vectors for the clusters and euclidiean distance
    cluster_assigned = np.zeros(dataset.shape[0], dtype=np.float64)
    distances = np.zeros([dataset.shape[0], k], dtype=np.float64)
    
    # Loop over the range of the datasets
    for i in range(len(dataset)):
        # Use euclidean distance to distribute point to the nearest centroid
        for i, c in enumerate(centroids):
            distances[:, i] = compute_euclidean_distance(c, dataset)
            
        # Place point in the right cluster based on closest centroid
        cluster_assigned = np.argmin(distances, axis=1)
        
        # Update centroid location using above cluster assignment
        for c in range(k):
            centroids[c] = np.mean(dataset[cluster_assigned == c], 0)
    return centroids, cluster_assigned

# Objective Function to return the result used in the lineplot
def objFunc(datapoints, maxK):
    # Loop a K range and append the min kmeans returned variables to the array before returning
    objFunc = []
    for i in range(1,maxK):
        centroids, cluster_assigned = kmeans(datapoints, i)
        objFunc.append(np.min(centroids))
    return objFunc

# Func to create the graph
def plotGraph(data, K, maxK):
    # Return the centroids/clusters for use below
    centroids, cluster_assigned = kmeans(data, K)
    
    # Begin figure creation. Plotting the datapoints from the file read in.
    plt.scatter(datapoints[:,0], datapoints[:,1], alpha=0.5)
    
    # Create the graph based on the amount of clusters required
    group_colors = ['skyblue', 'coral', 'lightgreen']
    colors = [group_colors[j] for j in cluster_assigned]
    plt.scatter(data[:,0], data[:,1], color=colors, alpha=0.5)
    
    # Create the centroids for the graph and use a specific colour depending on amount
    colorCentres=['blue', 'darkred', 'green']
    colorCentre = [colorCentres[cC] for cC in range(0,K)]
    plt.scatter(centroids[:,0], centroids[:,1], color=colorCentre)
    plt.ylabel('Length')
    plt.xlabel('Height')
    plt.show()
    
    #Objective Function line plot creation
    plt.plot(range(1,maxK), objFunc(data, maxK))
    plt.xlabel('Range of K')
    plt.ylabel('Objective Function Centroid Result')
    plt.show()
    
# Use the plot graph function to plot the specified data points, K number and the max K for the line plot
plotGraph(datapoints, 2, 10)