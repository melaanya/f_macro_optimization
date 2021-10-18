import unittest

from numba.typed import List
import numpy as np

from src.estimation import estimate_DV, format_output
from src.load import load


class EspGameNNMetrics(unittest.TestCase):
    """."""

    @classmethod
    def setUpClass(cls):  # noqa
        data = load("./data/ESPGameNN.txt")
        cls.data = List()
        [cls.data.append(x) for x in data]

    def test_1000_3x3(self):  # noqa
        answer = np.loadtxt("./data/hulls/result_ESPGameNN_1000_20.txt")

        output = estimate_DV(self.data, categCount=1000, angleCount=20)
        np.savetxt(
            "./data/outputs/python_ESPGameNN_1000_20.txt",
            output,
            delimiter=" ",
        )

        self.assertEqual(answer.shape, output.shape)
        self.assertTrue(
            np.allclose(answer, output, rtol=0, atol=1e-5, equal_nan=True)
        )
