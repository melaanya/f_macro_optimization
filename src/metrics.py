from typing import List, overload, Union

import numpy as np

from .classes import Categ


@overload
def f1(p: float, r: float) -> float:  # noqa
    ...


@overload
def f1(p: np.ndarray, r: np.ndarray) -> np.ndarray:  # noqa
    ...


# @jit(nopython=True)
def f1(
    p: Union[float, np.ndarray], r: Union[float, np.ndarray]
) -> Union[float, np.ndarray]:
    """Compute f score."""
    if (isinstance(p, float) and p == 0) or (isinstance(r, float) and r == 0):
        return 0

    if isinstance(p, np.ndarray) and np.allclose(p, 0):
        return np.zeros_like(p)

    if isinstance(r, np.ndarray) and np.allclose(r, 0):
        return np.zeros_like(r)

    return 2 * p * r / (p + r)


def precision_at_thr_k(data: List[Categ], k: int = 1) -> float:
    """Compute average precision at threshold = k."""
    tp: int = 0
    for categ in data:
        for ind, doc in enumerate(categ.pred):
            if doc == 1:
                tp += 1

            if ind + 1 == k:
                break
    p: float = tp / len(data) / k
    return p


def best_macro_f(data: List[Categ]) -> float:
    """Compute best macro-F score for data."""
    best_macro_f: float = 0
    for categ in data:
        tp: int = 0
        fp: int = 0
        best_macro_f_categ: float = -1e-6

        for doc in categ.pred:
            if doc == 1:
                tp += 1
            else:
                fp += 1

            p_k_thr: float = tp * 1.0 / (tp + fp)
            r_k_thr: float = tp * 1.0 / categ.size
            macro_f_thr = f1(p_k_thr, r_k_thr)
            if macro_f_thr > best_macro_f_categ:
                best_macro_f_categ = macro_f_thr

        best_macro_f += best_macro_f_categ
    best_macro_f /= len(data)
    return best_macro_f
