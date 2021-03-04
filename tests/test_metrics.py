import unittest

from src.load import load
from src.metrics import best_macro_f, precision_at_thr_k


class EspGameNNMetrics(unittest.TestCase):
    """Tests for ESP Game dataset NN prediction metrics computation."""

    @classmethod
    def setUpClass(cls):  # noqa
        cls.data = load("./data/ESPGameNN.txt")

    def test_p_1(self):  # noqa
        p_1 = precision_at_thr_k(self.data, k=1)
        self.assertAlmostEqual(p_1, 0.66791, delta=1e-5)

    def test_p_3(self):  # noqa
        p_3 = precision_at_thr_k(self.data, k=3)
        self.assertAlmostEqual(p_3, 0.589552, delta=1e-5)

    def test_p_5(self):  # noqa
        p_5 = precision_at_thr_k(self.data, k=5)
        self.assertAlmostEqual(p_5, 0.552985, delta=1e-5)

    def test_mean_f(self):  # noqa
        macro_f = best_macro_f(self.data)
        self.assertAlmostEqual(macro_f, 0.407092, delta=1e-5)
