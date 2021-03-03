from dataclasses import dataclass

import numpy as np


@dataclass
class Categ:
    """Category description."""

    id: int  # noqa
    size: int
    pred: np.ndarray
