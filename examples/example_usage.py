from gap_statistic.utils import sundar_tibshirani_gap_statistic_main
from sklearn.datasets import load_iris
import numpy as np

def main():
    iris = load_iris()
    X = iris.data
    y = iris.target  # These are the given labels for K=3

    k = 3
    B = 50
    pca_sampling = True
    standardize_within_pca = False
    distance_metric = 'euclidean'
    return_params = True
    use_user_labels = True  # Set to True to use user-given labels, False to apply KMeans for each B
    use_provided_labels = True  # Set to True to use provided labels, False to generate initial labels using KMeans
    n_init = 15
    random_state = 7142

    # Call the main function with the dataset and parameters
    sundar_tibshirani_gap_statistic_main(X, y, k, B=B, pca_sampling=pca_sampling, standardize_within_pca=standardize_within_pca, distance_metric=distance_metric, return_params=return_params, use_user_labels=use_user_labels, use_provided_labels=use_provided_labels, n_init=n_init, random_state=random_state)

if __name__ == "__main__":
    main()
