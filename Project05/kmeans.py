'''kmeans.py
Performs K-Means clustering
Scottie YANG Miaoyi
CS 251 Data Analysis Visualization, Fall 2021
'''
import numpy as np
import matplotlib.pyplot as plt
from palettable import cartocolors
from scipy.spatial import distance


class KMeans():
    def __init__(self, data=None):
        '''KMeans constructor

        (Should not require any changes)

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''

        # k: int. Number of clusters
        self.k = None
        # centroids: ndarray. shape=(k, self.num_features)
        #   k cluster centers
        self.centroids = None
        # data_centroid_labels: ndarray. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # data: ndarray. shape=(num_samps, num_features)
        self.data = data
        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None
        if data is not None:
            self.num_samps, self.num_features = data.shape

    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data
        self.num_samps, self.num_features = data.shape

    def get_data(self):
        '''Get a COPY of the data

        Returns:
        -----------
        ndarray. shape=(num_samps, num_features). COPY of the data
        '''
        return self.data.copy()

    def get_centroids(self):
        '''Get the K-means centroids

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(k, self.num_features).
        '''
        return self.centroids

    def get_data_centroid_labels(self):
        '''Get the data-to-cluster assignments

        (Should not require any changes)

        Returns:
        -----------
        ndarray. shape=(self.num_samps,)
        '''
        return self.data_centroid_labels

    def dist_pt_to_pt(self, pt_1, pt_2):
        '''Compute the Euclidean distance between data samples `pt_1` and `pt_2`

        Parameters:
        -----------
        pt_1: ndarray. shape=(num_features,)
        pt_2: ndarray. shape=(num_features,)

        Returns:
        -----------
        float. Euclidean distance between `pt_1` and `pt_2`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        euclidean_d = np.sum(np.linalg.norm(pt_1 - pt_2))
        return euclidean_d

    def dist_pt_to_centroids(self, pt, centroids):
        '''Compute the Euclidean distance between data sample `pt` and and all the cluster centroids
        self.centroids

        Parameters:
        -----------
        pt: ndarray. shape=(num_features,)
        centroids: ndarray. shape=(C, num_features)
            C centroids, where C is an int.

        Returns:
        -----------
        ndarray. shape=(C,).
            distance between pt and each of the C centroids in `centroids`.

        NOTE: Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running)
        '''
        euclidean_d = distance.cdist(pt.reshape(1, pt.shape[0]), centroids).flatten()
        return euclidean_d

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        self.k = k
        mins = np.amin(self.get_data(), axis = 0)
        maxs = np.amax(self.get_data(), axis = 0)

        self.centroids = np.random.uniform(low=mins, high=maxs, size=(k, self.num_features))
        return self.centroids

    def cluster(self, k=2, tol=1e-5, max_iter=1000, verbose=False):
        '''Performs K-means clustering on the data

        Parameters:
        -----------
        k: int. Number of clusters
        tol: float. Terminate K-means if the (absolute value of) the difference between all 
        the centroid values from the previous and current time step < `tol`.
        max_iter: int. Make sure that K-means does not run more than `max_iter` iterations.
        verbose: boolean. Print out debug information if set to True.

        Returns:
        -----------
        self.inertia. float. Mean squared distance between each data sample and its cluster mean
        int. Number of iterations that K-means was run for

        TODO:
        - Initialize K-means variables
        - Do K-means as long as the max number of iterations is not met AND the absolute value of the
        difference between the previous and current centroid values is > `tol`
        - Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        - Print out total number of iterations K-means ran for
        '''
        if verbose:
            print("debug;")
        else:
            self.initialize(k)

            iteration = 0
            previous_centroid = self.centroids
            self.data_centroid_labels = self.assign_labels(previous_centroid)

            while iteration < max_iter:
                previous_centroid = self.centroids
                self.centroids, centroid_diff = self.update_centroids(k, self.data_centroid_labels, previous_centroid)
                self.data_centroid_labels = self.assign_labels(self.centroids)
                self.inertia = self.compute_inertia()
                iteration += 1
                if -tol < np.mean(centroid_diff) < tol:
                    break
            return self.inertia, iteration

    def cluster_batch(self, k=2, n_iter=1, verbose=False):
        '''Run K-means multiple times, each time with different initial conditions.
        Keeps track of K-means instance that generates lowest inertia. Sets the following instance
        variables based on the best K-mean run:
        - self.centroids
        - self.data_centroid_labels
        - self.inertia

        Parameters:
        -----------
        k: int. Number of clusters
        n_iter: int. Number of times to run K-means with the designated `k` value.
        verbose: boolean. Print out debug information if set to True.
        '''
        lowest_centroids = None
        lowest_labels = None
        lowest_inertia = -1

        if verbose:
            print("debug;")
        else:
            for i in range(n_iter):
                self.cluster(k = k, verbose = verbose)
                if lowest_inertia == -1 or self.inertia < lowest_inertia:
                    lowest_inertia = self.inertia
                    lowest_labels = self.data_centroid_labels
                    lowest_centroids = self.centroids

        self.centroids = lowest_centroids
        self.data_centroid_labels = lowest_labels
        self.inertia = lowest_inertia


    def assign_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray. shape=(self.num_samps,). Holds index of the assigned cluster of each data sample

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        labels = []
        for i in range(self.num_samps):
            distances = self.dist_pt_to_centroids(self.get_data()[i], centroids)
            labels.append(np.argmin(distances))
        return np.array(labels)

    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values
        '''
        new_centroids = []
        location = []

        for i in range(k):
                                                                # in case of there're centroids that don't have data points in their clusters
            if isinstance(self.data_centroid_labels, type(None)) or (i in self.data_centroid_labels):
                for j in range(self.num_samps):
                    if data_centroid_labels[j] == i:
                        location.append(self.get_data()[j])
                location = np.array(location)
                new_centroid = np.mean(location, axis=0)
                location = []
                new_centroids.append(new_centroid)
            # in case of there're centroids that don't have data points in their clusters
            else:
                new_centroids.append(np.zeros(self.num_features))

        new_centroids = np.array(new_centroids)
        centroid_diff = new_centroids - prev_centroids

        return new_centroids, centroid_diff

    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Parameters:
        -----------
        None

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''
        centroids_match = np.empty((self.num_samps, self.num_features))
        for i in range(self.num_samps):
            centroids_match[i] = self.centroids[self.data_centroid_labels[i]]

        inertia = np.square(np.subtract(self.get_data(),centroids_match)).mean()*2

        return inertia

    def plot_clusters(self, figsize = (6.4, 4.8)):
        '''Creates a scatter plot of the data color-coded by cluster assignment.

        TODO:
        - Plot samples belonging to a cluster with the same color.
        - Plot the centroids in black with a different plot marker.
        - The default scatter plot color palette produces colors that may be difficult to discern
        (especially for those who are colorblind). Make sure you change your colors to be clearly
        differentiable.
            You should use a palette Colorbrewer2 palette. Pick one with a generous
            number of colors so that you don't run out if k is large (e.g. 10).
        '''
        plt.figure(figsize=figsize)
        color_list = cartocolors.qualitative.Vivid_10.mpl_colors
        for i in range(self.k):
            w = self.get_data()[self.data_centroid_labels == i]
            plt.scatter(w[:, 0], w[:, 1], color = color_list[i], label='cluster ' + str(i))

        #excluding centroids that don't have clusters
        centroids_that_work = []
        for j in range(self.k):
            if j in self.data_centroid_labels:
                centroids_that_work.append(j)
        plt.scatter(self.centroids[centroids_that_work,0], self.centroids[centroids_that_work,1], color = color_list[-1], label='centroids', marker = '*')
        
        plt.title('Data by ' + str(self.k) + ' clusters - inertia: ' + str(round(self.inertia,3)))
        plt.xlabel('x')
        plt.ylabel('y')
        plt.legend(bbox_to_anchor=(1, 0),loc='lower left', fontsize='small')

    def elbow_plot(self, max_k, n_iter = 1, figsize = (6.4, 4.8)):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.
        n_iter: int. Run k-means multipul times.

        TODO:
        - Run k-means with k=1,2,...,max_k, record the inertia.
        - Make the plot with appropriate x label, and y label, x tick marks.
        '''
        inertias = []
        for i in range(max_k):
            self.initialize(i+1)
            self.cluster_batch(i+1, n_iter = n_iter)
            inertia = self.compute_inertia()
            inertias.append(inertia)
        plt.figure(figsize=figsize)
        plt.plot(np.arange(1, max_k+1), inertias, 'o-')
        plt.xticks(np.arange(1, max_k+1))
        plt.xlabel('k clusters')
        plt.ylabel('Inertia')
        plt.title('The Elbow Plot showing inertias under different k')

    def replace_color_with_centroid(self):
        '''Replace each RGB pixel in self.data (flattened image) with the closest centroid value.
        Used with image compression after K-means is run on the image vector.

        Parameters:
        -----------
        None

        Returns:
        -----------
        None
        '''
        data = []
        for i in range(self.num_samps):
            centroid = np.rint(self.centroids[self.data_centroid_labels[i]]).astype(int)
            data.append(centroid)

        self.set_data(np.array(data))
