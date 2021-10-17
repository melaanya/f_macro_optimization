from typing import List

import numpy as np

from .classes import Categ


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
