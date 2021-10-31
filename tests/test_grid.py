import unittest

from numba.typed import List
import numpy as np

from src.estimation import estimate_grid
from src.io import format_output, load


class EspGameNNMetrics(unittest.TestCase):
    """."""

    @classmethod
    def setUpClass(cls):  # noqa
        data = load("./data/ESPGameNN.txt")
        cls.data = List()
        [cls.data.append(x) for x in data]

    def test_1000_3x3(self):  # noqa
        answer = np.loadtxt("./data/grids/result_ESPGameNN_1000_3x3.txt")

        n_p = 3
        res_p, res_r = estimate_grid(self.data, num_categs=1000, n_p=n_p)

        output = format_output(res_p, res_r, n_p)
        np.savetxt(
            "./data/outputs/python_ESPGameNN_1000_3x3.txt",
            output,
            delimiter=" ",
        )

        self.assertEqual(answer.shape, output.shape)
        self.assertTrue(
            np.allclose(answer, output, rtol=0, atol=1e-5, equal_nan=True)
        )

    def test_1000_10x10(self):  # noqa
        answer = np.loadtxt("./data/grids/result_ESPGameNN_1000_10x10.txt")

        n_p = 10
        res_p, res_r = estimate_grid(self.data, num_categs=1000, n_p=n_p)

        output = format_output(res_p, res_r, n_p)
        np.savetxt(
            "./data/outputs/python_ESPGameNN_1000_10x10.txt",
            output,
            delimiter=" ",
        )

        self.assertEqual(answer.shape, output.shape)

        mask = ~(np.isnan(answer) | np.isnan(output))
        self.assertTrue(
            np.allclose(answer[mask], output[mask], rtol=0, atol=1e-4)
        )

    def test_1000_6x6(self):  # noqa
        answer = np.loadtxt("./data/grids/result_ESPGameNN_1000_6x6.txt")

        n_p = 6
        res_p, res_r = estimate_grid(self.data, num_categs=1000, n_p=n_p)

        output = format_output(res_p, res_r, n_p)
        np.savetxt(
            "./data/outputs/python_ESPGameNN_1000_6x6.txt",
            output,
            delimiter=" ",
        )

        self.assertEqual(answer.shape, output.shape)
        self.assertTrue(
            np.allclose(answer, output, rtol=0, atol=1e-5, equal_nan=True)
        )

    def test_1000_9x9(self):  # noqa
        answer = np.loadtxt("./data/grids/result_ESPGameNN_1000_9x9.txt")

        n_p = 9
        res_p, res_r = estimate_grid(self.data, num_categs=1000, n_p=n_p)

        output = format_output(res_p, res_r, n_p)
        np.savetxt(
            "./data/outputs/python_ESPGameNN_1000_9x9.txt",
            output,
            delimiter=" ",
        )

        self.assertEqual(answer.shape, output.shape)
        self.assertTrue(
            np.allclose(answer, output, rtol=0, atol=1e-5, equal_nan=True)
        )
