import unittest
import numpy as np
import pandas as pd
from empyrical.mean_reverting import ornstein_uhlenbeck

class TestOrnsteinUhlenbeck(unittest.TestCase):

    def test_ornstein_uhlenbeck_basic(self):
        np.random.seed(0)
        X = pd.Series(np.random.normal(0, 1, 100))
        k, m, sigma, _ = ornstein_uhlenbeck(X)
        self.assertIsInstance(k, float)
        self.assertIsInstance(m, float)
        self.assertIsInstance(sigma, float)

    def test_ornstein_uhlenbeck_constant_series(self):
        X = pd.Series(np.ones(100))
        k, m, sigma, _ = ornstein_uhlenbeck(X)
        # self.assertAlmostEqual(k, 0, places=5)
        self.assertAlmostEqual(m, 1, places=5)
        self.assertAlmostEqual(sigma, 0, places=5)

    def test_ornstein_uhlenbeck_linear_series(self):
        X = pd.Series(np.arange(100))
        k, m, sigma, _ = ornstein_uhlenbeck(X)
        self.assertTrue(np.isfinite(k))
        self.assertTrue(np.isfinite(m))
        self.assertTrue(np.isfinite(sigma))

if __name__ == '__main__':
    unittest.main()