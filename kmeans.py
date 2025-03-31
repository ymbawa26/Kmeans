'''kmeans.py
Performs K-Means clustering
Yazan Bawaqna
CS 251/2: Data Analysis and Visualization
Spring 2025
'''
import numpy as np
import matplotlib.pyplot as plt


class KMeans:
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
        # data_centroid_labels: ndarray of ints. shape=(self.num_samps,)
        #   Holds index of the assigned cluster of each data sample
        self.data_centroid_labels = None

        self.labels = None

        # inertia: float.
        #   Mean squared distance between each data sample and its assigned (nearest) centroid
        self.inertia = None

        # num_samps: int. Number of samples in the dataset
        self.num_samps = None
        # num_features: int. Number of features (variables) in the dataset
        self.num_features = None

        if data is not None:
            self.set_data(data)  # Use set_data to initialize properly


    def set_data(self, data):
        '''Replaces data instance variable with `data`.

        Reminder: Make sure to update the number of data samples and features!

        Parameters:
        -----------
        data: ndarray. shape=(num_samps, num_features)
        '''
        self.data = data.copy()
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
        ndarray of ints. shape=(self.num_samps,)
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

        NOTE:
        - Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running).
        - Implement the distance formula (see notebook), do not use np.linalg.norm here.
        '''
        return np.sqrt(np.sum((pt_1 - pt_2) ** 2))

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

        NOTE:
        - Implement without any for loops (you will thank yourself later since you will wait
        only a small fraction of the time for your code to stop running).
        - Implement the distance formula (see notebook), do not use np.linalg.norm here.
        '''
            # Step 1: Calculate the difference between pt and each centroid (broadcasting)
        differences = centroids - pt

        # Step 2: Square each element in the differences
        squared_differences = differences ** 2

        # Step 3: Sum the squared differences along each centroid (sum along axis 1)
        sum_of_squares = np.sum(squared_differences, axis=1)

        # Step 4: Take the square root to get the Euclidean distance
        distances = np.sqrt(sum_of_squares)

        # Step 5: Return the computed distances
        return distances

    def initialize(self, k):
        '''Initializes K-means by setting the initial centroids (means) to K unique randomly
        selected data samples and inertia to infinity (i.e. np.inf).

        Parameters:
        -----------
        k: int. Number of clusters

        Returns:
        -----------
        ndarray. shape=(k, self.num_features). Initial centroids for the k clusters.

        NOTE: Can be implemented without any for loops
        '''
        # Step 1: Set the number of clusters (k)
        self.k = k

        # Step 2: Randomly select k unique indices from the dataset
        random_indices = np.random.choice(self.num_samps, k, replace=False)

        # Step 3: Select the data points at those indices as initial centroids
        self.centroids = self.data[random_indices]

        # Step 4: Set inertia to infinity
        self.inertia = np.inf

        # Step 5: Return the initial centroids
        return self.centroids

    def cluster(self, k=2, tol=1e-2, max_iter=1000, verbose=False, p=2):
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
        1. Initialize K-means variables
        2. Do K-means as long as the max number of iterations is not met AND the absolute value of the difference between
        the previous and current inertia is bigger than the tolerance `tol`. K-means should always run for at least 1
        iteration.
        3. Set instance variables based on computed values.
        (All instance variables defined in constructor should be populated with meaningful values)
        4. Print out total number of iterations K-means ran for
        '''
        self.initialize(k)
        prev_inertia = np.inf
        for iteration in range(max_iter):
            distances = np.array([self.dist_pt_to_centroids(pt, self.centroids) for pt in self.data])
            self.data_centroid_labels = np.argmin(distances, axis=1)

            new_centroids = np.array([self.data[self.data_centroid_labels == i].mean(axis=0) if np.any(self.data_centroid_labels == i) else self.centroids[i] for i in range(k)])

            self.inertia = np.mean([self.dist_pt_to_pt(self.data[i], self.centroids[self.data_centroid_labels[i]]) ** 2 for i in range(self.num_samps)])

            if np.abs(prev_inertia - self.inertia) < tol:
                if verbose:
                    print(f'Converged in {iteration + 1} iterations.')
                break
            
            self.centroids = new_centroids
            prev_inertia = self.inertia

        return self.inertia, iteration + 1

    def cluster_batch(self, k=2, n_iter=1, verbose=False, p=2):
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
        best_inertia = np.inf
        best_centroids = None
        best_labels = None

        for _ in range(n_iter):
            self.cluster(k, verbose=verbose, p=p)
            if self.inertia < best_inertia:
                best_inertia = self.inertia
                best_centroids = self.centroids.copy()
                best_labels = self.data_centroid_labels.copy()

        self.centroids = best_centroids
        self.data_centroid_labels = best_labels
        self.inertia = best_inertia


    def update_labels(self, centroids):
        '''Assigns each data sample to the nearest centroid

        Parameters:
        -----------
        centroids: ndarray. shape=(k, self.num_features). Current centroids for the k clusters.

        Returns:
        -----------
        ndarray of ints. shape=(self.num_samps,). Holds index of the assigned cluster of each data
            sample. These should be ints (pay attention to/cast your dtypes accordingly).

        Example: If we have 3 clusters and we compute distances to data sample i: [0.1, 0.5, 0.05]
        labels[i] is 2. The entire labels array may look something like this: [0, 2, 1, 1, 0, ...]
        '''
        distances = np.array([self.dist_pt_to_centroids(pt, centroids) for pt in self.data])
        return np.argmin(distances, axis=1).astype(int)


    def update_centroids(self, k, data_centroid_labels, prev_centroids):
        '''Computes each of the K centroids (means) based on the data assigned to each cluster

        Parameters:
        -----------
        k: int. Number of clusters
        data_centroid_labels. ndarray of ints. shape=(self.num_samps,)
            Holds index of the assigned cluster of each data sample
        prev_centroids. ndarray. shape=(k, self.num_features)
            Holds centroids for each cluster computed on the PREVIOUS time step

        Returns:
        -----------
        new_centroids. ndarray. shape=(k, self.num_features).
            Centroids for each cluster computed on the CURRENT time step
        centroid_diff. ndarray. shape=(k, self.num_features).
            Difference between current and previous centroid values

        NOTE: Your implementation should handle the case when there are no samples assigned to a cluster —
        i.e. `data_centroid_labels` does not have a valid cluster index in it at all.
            For example, if `k`=3 and data_centroid_labels = [0, 1, 0, 0, 1], there are no samples assigned to cluster 2.
        In the case of each cluster without samples assigned to it, you should assign make its centroid a data sample
        randomly selected from the dataset.
        '''
        """Computes new centroids based on assigned data points."""
    
        # Step 1: Initialize an empty array for new centroids
        new_centroids = np.zeros((k, self.num_features))

        # Step 2: Loop through each cluster index
        for cluster_index in range(k):
            # Step 3: Find all data points assigned to this cluster
            cluster_points = self.data[data_centroid_labels == cluster_index]

            # Step 4: Compute new centroid
            if len(cluster_points) > 0:  
            # If the cluster has points, compute the mean of all points in the cluster
                new_centroids[cluster_index] = np.mean(cluster_points, axis=0)
            else:
                # If the cluster has no points, randomly assign one data point as the centroid
                random_index = np.random.choice(self.num_samps)
                new_centroids[cluster_index] = self.data[random_index]

        # Step 5: Compute difference between new and previous centroids
        centroid_diff = new_centroids - prev_centroids

        return new_centroids, centroid_diff


    def compute_inertia(self):
        '''Mean squared distance between every data sample and its assigned (nearest) centroid

        Returns:
        -----------
        float. The average squared distance between every data sample and its assigned cluster centroid.
        '''

    
        # Step 1: Initialize total squared distance
        total_squared_distance = 0.0

        # Step 2: Iterate through each data point
        for i in range(self.num_samps):
            # Step 3: Get the assigned centroid index for the data point
            assigned_centroid_idx = self.data_centroid_labels[i]

            # Step 4: Compute squared distance from data point to its centroid
            squared_distance = self.dist_pt_to_pt(self.data[i], self.centroids[assigned_centroid_idx]) ** 2

            # Step 5: Accumulate squared distances
            total_squared_distance += squared_distance

            # Step 6: Compute mean squared distance (inertia)
            inertia = total_squared_distance / self.num_samps

        return inertia


    def elbow_plot(self, max_k):
        '''Makes an elbow plot: cluster number (k) on x axis, inertia on y axis.

        Parameters:
        -----------
        max_k: int. Run k-means with k=1,2,...,max_k.

        '''
        inertias = []  # List to store the inertia for each k
        k_values = range(1, max_k + 1)  # Range of k values from 1 to max_k

        for k in k_values:
            self.cluster(k)  # Run k-means with k clusters
            inertias.append(self.inertia)  # Record the inertia

        # Plot the elbow plot
        plt.plot(k_values, inertias, marker='o')
        plt.xlabel('Number of clusters (k)')
        plt.ylabel('Inertia (Within-cluster sum of squares)')
        plt.title('Elbow Plot')
        plt.xticks(k_values)  # Set x-ticks to match the k values
        plt.grid(True)
        plt.show()



    def plot_clusters(self):
        """Creates a scatter plot of the data color-coded by cluster assignment.

        - Samples in the same cluster share the same color.
        - Centroids are plotted in black with a distinct marker.
        - Uses the Okabe-Ito colorblind-friendly palette.
        """

        import matplotlib.pyplot as plt

        # Step 1: Define a colorblind-friendly palette
        colors = [
        "#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2", 
        "#D55E00", "#CC79A7", "#999999"
        ]
    
        # Step 2: Ensure enough colors for all clusters
        num_colors_needed = max(self.k, len(colors))
        cluster_colors = colors * (num_colors_needed // len(colors) + 1)

        # Step 3: Create the scatter plot
        plt.figure(figsize=(8, 6))
        for cluster_idx in range(self.k):
            # Select data points belonging to this cluster
            cluster_points = self.data[self.data_centroid_labels == cluster_idx]
            plt.scatter(cluster_points[:, 0], cluster_points[:, 1], 
                    color=cluster_colors[cluster_idx], label=f'Cluster {cluster_idx}', alpha=0.6)

        # Step 4: Plot the centroids in black with distinct markers
        plt.scatter(self.centroids[:, 0], self.centroids[:, 1], 
                color='black', marker='X', s=150, label='Centroids')

        # Step 5: Add labels and title
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
        plt.title("K-Means Clustering Visualization")
        plt.legend()
        plt.grid(True)
        plt.show()

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
        # Step 1: Ensure clustering has been run
        if self.centroids is None or self.data_centroid_labels is None:
            raise ValueError("K-Means clustering must be run before replacing colors.")

        # Step 2: Replace each pixel with the centroid it was assigned to
        for i in range(self.num_samps):
            cluster_idx = self.data_centroid_labels[i]  # Get assigned cluster index
            self.data[i] = self.centroids[cluster_idx]  # Replace pixel with centroid color
