from typing import List

from .classes import Categ


def f1(p: float, r: float) -> float:
    """Compute f score."""
    if p == 0 or r == 0:
        return 0
    else:
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
