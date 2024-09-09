# gap_statistic/gap_stat.py
import numpy as np
from typing import Callable, Union, Tuple
from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from scipy.linalg import svd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import pandas as pd
import matplotlib.pyplot as plt

class SundarTibshiraniGapStatistic:
    """
    Implements the Sundar-Tibshirani Gap Statistic for cluster analysis.
    
    This class provides methods to calculate the Gap Statistic for a given
    cluster assignment, which can be used to evaluate the clustering solution.
    """

    def __init__(self,
                 distance_metric: Union[str, Callable] = 'euclidean',
                 pca_sampling: bool = True,
                 standardize_within_pca: bool = False,
                 return_params: bool = False,
                 use_user_labels: bool = True,
                 n_init: int = 12,
                 random_state: int = 7142):
        """
        Initialize the SundarTibshiraniGapStatistic object.

        Args:
            distance_metric (str or callable): The distance metric to use. Options are 'euclidean', 'minkowski', 'manhattan', 'cosine', 'l1', 'l2'.
            pca_sampling (bool): Whether to use PCA for reference distribution sampling. Default is True.
            standardize_within_pca (bool): Whether to standardize data within PCA. Default is False.
            return_params (bool): Whether to return additional parameters. Default is False.
            use_user_labels (bool): Whether to use user-given labels for each reference dataset. If False, KMeans is applied to each reference dataset. Default is True.
            n_init (int): Number of time the k-means algorithm will be run with different centroid seeds. Default is 12.
            random_state (int): Determines random number generation for centroid initialization. Default is 7142.
        """
        self.valid_metrics = ['euclidean', 'minkowski', 'manhattan', 'cosine', 'l1', 'l2']
        if isinstance(distance_metric, str):
            if distance_metric not in self.valid_metrics:
                raise ValueError(f"Invalid distance metric. Choose from {self.valid_metrics}")
            self.distance_metric = lambda X, Y: pairwise_distances(X, Y, metric=distance_metric)
        elif callable(distance_metric):
            self.distance_metric = distance_metric
        else:
            raise TypeError("distance_metric must be either a string or a function")
        
        self.pca_sampling = pca_sampling
        self.standardize_within_pca = standardize_within_pca
        self.return_params = return_params
        self.use_user_labels = use_user_labels
        self.n_init = n_init
        self.random_state = random_state

    def _calculate_within_cluster_dispersion(self, X: np.ndarray, labels: np.ndarray) -> float:
        """
        Calculate the within-cluster dispersion.

        Args:
            X (np.ndarray): The input data.
            labels (np.ndarray): The cluster labels.

        Returns:
            float: The calculated Wk value.
        """
        unique_labels = np.unique(labels)
        centroids = np.array([X[labels == i].mean(axis=0) for i in unique_labels])

        Wk = 0
        for i in unique_labels:
            cluster_points = X[labels == i]
            distances = self.distance_metric(cluster_points, centroids[i].reshape(1, -1))
            Wk += np.sum(distances)

        Wk /= (2 * len(X))
        return Wk

    def _simulate_within_cluster_dispersions(self, X: np.ndarray, labels: np.ndarray, k: int, B: int) -> np.ndarray:
        """
        Simulate Wk values for reference distributions.

        Args:
            X (np.ndarray): The input data.
            labels (np.ndarray): The cluster labels.
            k (int): The number of clusters to use for KMeans.
            B (int): The number of iterations for simulation.

        Returns:
            np.ndarray: An array of simulated Wk values.
        """
        if self.pca_sampling:
            print("PCA Sampling is Enabled.")
            if self.standardize_within_pca:
                print("Standardizing within PCA.")
                scaler = StandardScaler()
                scaled_X = scaler.fit_transform(X)
            else:
                print("NOT Standardizing within PCA.")
                scaled_X = X
            _, _, VT = svd(scaled_X)
            X_prime = np.dot(X, VT.T)
        else:
            print("PCA Sampling is Disabled.")
            X_prime = X

        simulated_Wks = []

        for _ in range(B):
            Z_prime = np.random.uniform(np.min(X_prime), np.max(X_prime), size=(len(X), X.shape[1]))

            if self.pca_sampling:
                sampled_X = np.dot(Z_prime, VT)
            else:
                sampled_X = Z_prime

            if self.use_user_labels:
                Wk_star = self._calculate_within_cluster_dispersion(sampled_X, labels)
            else:
                kmeans = KMeans(n_clusters=k, n_init=self.n_init, random_state=self.random_state)
                kmeans.fit(sampled_X)
                Wk_star = self._calculate_within_cluster_dispersion(sampled_X, kmeans.labels_)

            simulated_Wks.append(Wk_star)

        return np.array(simulated_Wks)

    def compute_gap_statistic(self, X: np.ndarray, labels: np.ndarray, k: int, B: int = 20) -> Union[float, Tuple[float, dict]]:
        """
        Compute the Gap Statistic for the given cluster assignment.

        Args:
            X (np.ndarray): The input data.
            labels (np.ndarray): The cluster labels.
            k (int): The number of clusters to use for KMeans.
            B (int): The number of iterations for simulation. Default is 20.

        Returns:
            float or tuple: The calculated Gap Statistic, or a tuple of the Gap Statistic and additional parameters.

        Raises:
            TypeError: If input types are incorrect.
            ValueError: If B is too large.
        """
        if not isinstance(k, int):
            raise TypeError('k must be of type int')
        # modify the compute_gap_statistic method to raise an error if the number of unique labels is less than k (similar to the check you already have for k > len(np.unique(labels))).
        if k > len(np.unique(labels)):
            raise ValueError(f"The number of clusters k ({k}) cannot be larger than the number of unique labels ({len(np.unique(labels))})")
        # The number of clusters k ({k}) cannot be larger than the number of unique labels ({len(np.unique(labels))})
        if k > len(np.unique(labels)):
            raise ValueError(f"The number of clusters k ({k}) cannot be larger than the number of unique labels ({len(np.unique(labels))})")
        if not isinstance(B, int):
            raise TypeError('B must be of type int')
        if B > 500:
            raise ValueError('B is too big, B must be less than or equal to 500')
        if isinstance(X, list):
            X = np.array(X)
        elif not isinstance(X, np.ndarray):
            raise TypeError('Please provide either a list or a numpy array for X')

        Wk = self._calculate_within_cluster_dispersion(X, labels)
        sim_Wks = self._simulate_within_cluster_dispersions(X, labels, k, B)

        log_Wk = np.log(Wk)
        log_sim_Wks = np.log(sim_Wks)

        gap = np.mean(log_sim_Wks) - log_Wk
        sd_k = np.std(log_sim_Wks)
        sim_sks = np.sqrt(1 + (1 / B)) * sd_k

        if self.return_params:
            params = {'Wk': Wk, 'sim_Wks': sim_Wks, 'sim_sks': sim_sks, 'gap': gap, 'sd_k': sd_k}
            return gap, params
        else:
            return gap
