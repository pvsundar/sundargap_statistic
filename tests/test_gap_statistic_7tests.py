import unittest
from gap_statistic.gap_stat import SundarTibshiraniGapStatistic
import numpy as np

class TestGapStatistic(unittest.TestCase):

    def test_compute_gap_statistic(self):
        # Example data and labels
        X = np.random.rand(100, 5)
        labels = np.random.randint(0, 3, 100)
        k = 3
        B = 20

        gap_stat = SundarTibshiraniGapStatistic()
        gap = gap_stat.compute_gap_statistic(X, labels, k, B)
        self.assertIsInstance(gap, float)

    def test_invalid_distance_metric(self):
        with self.assertRaises(ValueError):
            SundarTibshiraniGapStatistic(distance_metric='invalid_metric')

    def test_invalid_B(self):
        with self.assertRaises(ValueError):
            SundarTibshiraniGapStatistic().compute_gap_statistic(np.random.rand(100, 5), np.random.randint(0, 3, 100), 3, B=501)

    # Additional test cases:
    
    # 1. Test with Non-Numerical Data
    def test_non_numerical_data(self):
        X = [['a', 'b'], ['c', 'd']]  # Non-numerical data
        labels = np.array([0, 1])
        with self.assertRaises(TypeError):
            SundarTibshiraniGapStatistic().compute_gap_statistic(X, labels, k=2)

    # 2. Test with Empty Data
    def test_empty_data(self):
        X = np.array([]).reshape(0, 5)  # Empty data with 5 features
        labels = np.array([])
        with self.assertRaises(ValueError):
            SundarTibshiraniGapStatistic().compute_gap_statistic(X, labels, k=3)

    # 3. Test for Large `k` (More clusters than data points)
    def test_large_k(self):
        X = np.random.rand(10, 5)  # 10 data points
        labels = np.random.randint(0, 2, 10)  # 2 clusters
        with self.assertRaises(ValueError):
            SundarTibshiraniGapStatistic().compute_gap_statistic(X, labels, k=20)  # 20 clusters (too large)

    # 4. Test with Different Distance Metrics
    def test_different_distance_metrics(self):
        X = np.random.rand(100, 5)
        labels = np.random.randint(0, 3, 100)
        for metric in ['euclidean', 'manhattan', 'cosine']:
            gap_stat = SundarTibshiraniGapStatistic(distance_metric=metric)
            gap = gap_stat.compute_gap_statistic(X, labels, k=3)
            self.assertIsInstance(gap, float)

if __name__ == '__main__':
    unittest.main(exit=False)

