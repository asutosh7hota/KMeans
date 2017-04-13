#! /usr/bin/env python

"""kmeans.py: Processes the KMeans algorithm for given dataset and value of k."""

# imports
import numpy as np
import scipy.spatial

def find_means(observations, clusters, k):
    """Return updated means of clusters.

    observations is the array of our vector observations.

    clusters is an array of integers that represent the indices
    of the cluster to which each observation belongs.

    k is the number of specified clusters.

    The output is an array containing each cluster's updated mean.

    """
    # Create uninitialized array
    output = np.empty(shape=(k,) + observations.shape[1:])

    # Initialize with the updated means
    for i in range(k):
        np.mean(observations[clusters == i], axis=0, out=output[i])
    return output



def kmeans(observations, k):
    """Divide the observations into clusters using the k-means
    algorithm, and return an array of integers assigning each observations
    point to one of the clusters.

    observations is the array of our vector observations.

    k gives the number of clusters and the initial positions of the 
    means are selected randomly from the observations.

    The output is a string representation of a sorted (ascending order) 
    list that contains the counts of how many observations are in each
    of the k clusters.

    """
    # Initialization of cluster means by randomly choosing k observations (Forgy Method)
    means = observations[np.random.choice(np.arange(len(observations)), k, False)]

    # Iterate until the previous cluster means are identical to new cluster means (convergence)
    while (True):
        # Squared euclidean distances between each obs and each cluster mean.
        sq_euclidean_dist = scipy.spatial.distance.cdist(means, observations, 'sqeuclidean')

        # Indices of the closest mean to each observation (defines current clusters)
        clusters = np.argmin(sq_euclidean_dist, axis=0)

        # Re-calculate cluster means
        new_means = find_means(observations, clusters, k)

        # Stop if no means were updated (convergence)
        if np.array_equal(new_means, means):
            break

        means = new_means
    output = sorted([clusters.tolist().count(x) for x in set(clusters)])
    return str(output)