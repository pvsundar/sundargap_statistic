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

if __name__ == '__main__':
    unittest.main(exit=False)
