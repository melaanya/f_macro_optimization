from numba import int32, int8
from numba.experimental import jitclass
import numpy as np


spec = [
    ("id", int32),
    ("size", int32),
    ("pred", int8[:]),
]


@jitclass(spec)
class Categ:
    """Category description."""

    def __init__(self, id: int, size: int, pred: np.ndarray):  # noqa
        self.id = id  # noqa
        self.size = size
        self.pred = pred
