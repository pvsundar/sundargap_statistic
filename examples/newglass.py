# `newglass.py` file  
#Glass dataset using the Sundar-Tibshirani Gap Statistic. 
#This script retains all the comments and options for flexibility.

# newglass.py

from gap_statistic.utils import sundar_tibshirani_gap_statistic_main
from sklearn.datasets import fetch_openml
import numpy as np

if __name__ == "__main__":
    # Load the Glass dataset from OpenML
    glass_data = fetch_openml(name='glass', version=1, as_frame=True, parser='auto')

    # Extract features and labels
    X = glass_data.data
    y = glass_data.target.astype('category').cat.codes  # Convert target to integer codes for clustering

    # Determine the number of unique labels
    k = len(np.unique(y))
    B = 100  # Number of iterations for simulation
    pca_sampling = True  # Whether to use PCA sampling
    standardize_within_pca = False  # Whether to standardize data within PCA
    distance_metric = 'euclidean'  # Distance metric to use
    return_params = True  # Whether to return additional parameters
    ## FOR THE CORRECT NULL HYPOTHESIS WHEN THE SOLUTION IS KNOWN / SPECIFIED:
    ## USE use_user_labels = True; use_provided_labels = True
    ## *TRUE* AND *TRUE* FOR THE CORRECT NULL HYPOTHESIS
    ## Set to True to use user-given labels,
    ## Set to True to use provided labels
    use_user_labels = True  # Set to True to use user-given labels, False to apply KMeans for each B
    use_provided_labels = True  # Set to True to use provided labels, False to generate initial labels using KMeans
    
    # FOR THE ORIGINAL NULL HYPOTHESIS:
    ## FOR CASES WHERE WE DONT KNOW THE USER LABELS WE USE KMEANS; AND, WE WANT TO USE KMEANS FOR EACH B.
    #use_user_labels = False  # Set to True to use user-given labels, False to apply KMeans for each B
    #use_provided_labels = False  # Set to True to use provided labels, False to generate initial labels using KMeans
    
    n_init = 15  # Number of times the K-Means algorithm will be run with different centroid seeds
    random_state = 7142  # Determines random number generation for centroid initialization

    # Call the main function with the dataset and parameters
    sundar_tibshirani_gap_statistic_main(X, y, k, B=B, pca_sampling=pca_sampling, standardize_within_pca=standardize_within_pca, distance_metric=distance_metric, return_params=return_params, use_user_labels=use_user_labels, use_provided_labels=use_provided_labels, n_init=n_init, random_state=random_state)


# This script loads the Glass dataset, extracts features and labels, and then calls the `sundar_tibshirani_gap_statistic_main` function with the specified parameters to compute the Sundar-Tibshirani Gap Statistic. You can adjust the parameters as needed to suit different scenarios.
