# Simulation for All Variants (True/False Combinations)
# Running all possible combinations of the use_user_labels and use_provided_labels parameters will provide a complete view of how the algorithm behaves in different scenarios. This will be particularly useful for comparing cases where:
# The user knows the true solution.
# The algorithm needs to generate cluster assignments itself.

from gap_statistic.utils import sundar_tibshirani_gap_statistic_main
from sklearn.datasets import fetch_openml, load_iris
import numpy as np

def run_simulation(X, y, k, dataset_name):
    B = 50
    pca_sampling = True
    standardize_within_pca = False
    distance_metric = 'euclidean'
    n_init = 15
    random_state = 7142

    # Test all combinations of user-provided and algorithm-generated labels
    combinations = [(True, True), (True, False), (False, True), (False, False)]
    
    for use_user_labels, use_provided_labels in combinations:
        print(f"\nRunning simulation for {dataset_name} with:")
        print(f"use_user_labels = {use_user_labels}, use_provided_labels = {use_provided_labels}")

        # Call the main function with the dataset and parameters
        sundar_tibshirani_gap_statistic_main(X, y, k, B=B, pca_sampling=pca_sampling,
                                             standardize_within_pca=standardize_within_pca,
                                             distance_metric=distance_metric, return_params=True,
                                             use_user_labels=use_user_labels, use_provided_labels=use_provided_labels,
                                             n_init=n_init, random_state=random_state)

if __name__ == "__main__":
    # Load Iris dataset
    iris = load_iris()
    X_iris = iris.data
    y_iris = iris.target
    k_iris = 3
    run_simulation(X_iris, y_iris, k_iris, "Iris")

    # Load Glass dataset
    glass_data = fetch_openml(name='glass', version=1, as_frame=True, parser='auto')
    X_glass = glass_data.data
    y_glass = glass_data.target.astype('category').cat.codes
    k_glass = len(np.unique(y_glass))
    run_simulation(X_glass, y_glass, k_glass, "Glass")
