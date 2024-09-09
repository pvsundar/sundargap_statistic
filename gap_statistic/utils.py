# gap_statistic/utils.py

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
#from .utils import sundar_tibshirani_gap_statistic_main
from gap_statistic.gap_stat import SundarTibshiraniGapStatistic # Import our class here

# Define the SundarTibshiraniGapStatistic class
def sundar_tibshirani_gap_statistic_main(X, labels=None, k: int = 3, B: int = 50, 
                                         pca_sampling=True, standardize_within_pca=False, 
                                         distance_metric='euclidean', return_params=False, 
                                         use_user_labels=True, use_provided_labels=True, 
                                         n_init=12, random_state=7142):
    """
    Main function to compute the Sundar-Tibshirani Gap Statistic for a given dataset and cluster labels.

    Parameters:
    - X (np.ndarray): The data points.
    - labels (np.ndarray): The cluster labels for the data points.
    - k (int): The number of clusters to use for KMeans.
    - B (int): The number of iterations for simulation.
    - pca_sampling (bool): Whether to use PCA sampling.
    - standardize_within_pca (bool): Whether to standardize data within PCA.
    - distance_metric (str): The distance metric to use.
    - return_params (bool): Whether to return additional parameters.
    - use_user_labels (bool): Whether to use user-given labels for each reference dataset.
    - use_provided_labels (bool): Whether to use provided labels for initial clustering.
    - n_init (int): Number of times the K-Means algorithm will be run with different centroid seeds. Default is 12.
    - random_state (int): Determines random number generation for centroid initialization. Default is 7142.
    """
    
    # Scale the data using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Initialize the Sundar-Tibshirani Gap Statistic class
    gap_stat = SundarTibshiraniGapStatistic(
        pca_sampling=pca_sampling, 
        standardize_within_pca=standardize_within_pca, 
        distance_metric=distance_metric, 
        return_params=return_params, 
        use_user_labels=use_user_labels, 
        n_init=n_init, 
        random_state=random_state
    )

    # If no labels are provided, generate them using KMeans
    if not use_provided_labels or labels is None:
        print("Generating initial labels using KMeans.")
        kmeans = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
        labels = kmeans.fit_predict(X_scaled)

    # Compute the Gap Statistic and return results
    if return_params:
        gap, params = gap_stat.compute_gap_statistic(X_scaled, labels, k, B=B)
        print(f"Gap Statistic for given clustering: {gap}")
        print(f"Value of correct k specified: {k}")
        print(f"Standard Deviation: {params['sd_k']}")
        print(f"Average simulated Wk: {np.mean(params['sim_Wks'])}")
        print(f"Method of Ho: {'User-given labels' if use_user_labels else 'KMeans clustering'}")
    else:
        gap = gap_stat.compute_gap_statistic(X_scaled, labels, k, B=B)
        print(f"Gap Statistic for given clustering: {gap}")
        print(f"Method of Ho: {'User-given labels' if use_user_labels else 'KMeans clustering'}")
