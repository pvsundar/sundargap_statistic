from sklearn.datasets import load_iris
from gap_statistic.utils import sundar_tibshirani_gap_statistic_main
from sklearn.datasets import fetch_openml
import numpy as np
X, y = load_iris(return_X_y=True)
sundar_tibshirani_gap_statistic_main(X, y, k=3)
