from typing import Tuple

import numpy as np

from .classes import Categ
from .metrics import f_beta


def load(filename: str):
    """Load and parse the file with predictions."""
    categ_list = []
    with open(filename, "r") as fp:
        cur_line = fp.readline().strip()
        while cur_line:
            cur_elements = cur_line.split(" ")
            cur_id, cur_size = int(cur_elements[0]), int(cur_elements[1])
            pred = np.array(cur_elements[2:], dtype=np.uint8)
            cur_categ = Categ(cur_id, cur_size, pred)
            categ_list.append(cur_categ)

            cur_line = fp.readline().strip()

    return categ_list


def format_output(
    res_p: np.ndarray,
    res_r: np.ndarray,
    n_p: int,
    beta: float = 1.0,
    left_bottom: Tuple[float, float] = (0.0, 0.0),
    right_top: Tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Format grid in expected format for further plotting."""
    p_0, r_0 = left_bottom
    p_1, r_1 = right_top
    n_r = n_p

    output = np.zeros(((n_p + 1) * (n_r + 1), 9), dtype=np.float64)
    for i_p in range(n_p + 1):
        for i_r in range(n_r + 1):
            output[i_p * (n_p + 1) + i_r, 0] = p_0 + (p_1 - p_0) / n_p * i_p
            output[i_p * (n_p + 1) + i_r, 1] = r_0 + (r_1 - r_0) / n_r * i_r

    output[:, 2] = res_p[1].flatten()
    output[:, 3] = res_r[1].flatten()

    output[:, 4] = f_beta(output[:, 2], output[:, 3], beta)

    output[:, 5] = res_p[0].flatten()  # min
    output[:, 6] = res_p[2].flatten()  # max

    output[:, 7] = res_r[0].flatten()  # min
    output[:, 8] = res_r[2].flatten()  # max

    return output