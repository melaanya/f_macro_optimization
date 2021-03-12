import unittest

import numpy as np

from src.estimation import estimate_grid, format_output
from src.load import load


class EspGameNNMetrics(unittest.TestCase):
    """."""

    @classmethod
    def setUpClass(cls):  # noqa
        cls.data = load("./data/ESPGameNN.txt")

    def test_1000_3x3(self):  # noqa
        answer = np.loadtxt("./data/grids/result_ESPGameNN_1000_3x3.txt")

        n_p = 3
        res_p, res_r = estimate_grid(self.data, num_categs=1000, n_p=n_p)

        output = format_output(res_p, res_r, n_p)
        np.savetxt(
            "./data/outputs/python_ESPGameNN_1000_3x3.txt",
            output,
            delimiter=" ",
            fmt="%10.10f",
        )

        self.assertEqual(answer.shape, output.shape)
        self.assertTrue(
            np.allclose(answer, output, rtol=0, atol=1e-5, equal_nan=True)
        )

    # def test_1000_10x10(self):
    #     answer = np.loadtxt("./data/grids/result_ESPGameNN_1000_10x10.txt")

    #     n_p = 10
    #     res_p, res_r = estimate_grid(self.data, num_categs=1000, n_p=n_p)

    #     output = format_output(res_p, res_r, n_p)
    #     np.savetxt(
    #         "./data/outputs/python_ESPGameNN_1000_10x10.txt",
    #         output,
    #         delimiter=" ",
    #         fmt="%10.10f",
    #     )

    #     self.assertEqual(answer.shape, output.shape)
    #     self.assertTrue(
    #         np.allclose(answer, output, rtol=0, atol=1e-5, equal_nan=True)
    #     )
