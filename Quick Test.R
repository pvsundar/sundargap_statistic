library(reticulate)

py_run_string("
from gap_statistic.gap_stat import SundarTibshiraniGapStatistic
from gap_statistic.utils import sundar_tibshirani_gap_statistic_main
from sklearn.datasets import load_iris
import numpy as np

# Load the dataset
iris = load_iris()
X = iris.data
y = iris.target

# Run the function
sundar_tibshirani_gap_statistic_main(X, y, k=3)
")
