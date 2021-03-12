from typing import List, Tuple

from numba import jit
import numpy as np

from .classes import Categ
from .metrics import f1


@jit(nopython=True)
def estimate_grid(
    data: List[Categ], num_categs: int, n_p: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimating F-macro in every point of a unit square grid,
    the number of points in grid is n_p * n_p.
    """
    num_categs = len(data) if len(data) < num_categs else num_categs

    n_r = n_p  # the same amount of points in x and y axis
    p_0, p_1, r_0, r_1 = 0.0, 1.0, 0.0, 1.0

    res_p = np.zeros((3, n_p + 1, n_r + 1), dtype=np.float64)
    res_r = np.zeros((3, n_p + 1, n_r + 1), dtype=np.float64)

    # with tqdm(total=(n_p + 1) * (n_r + 1)) as pbar:
    for i_p in range(n_p + 1):
        for i_r in range(n_r + 1):
            map_0 = p_0 + (p_1 - p_0) / n_p * i_p
            mar_0 = r_0 + (r_1 - r_0) / n_r * i_r

            alpha, beta = mar_0 ** 2, map_0 ** 2
            delta = 1.0 / num_categs

            eps = (
                2
                * delta ** 2
                * (map_0 + mar_0 + 2 * delta) ** 2
                / (map_0 + mar_0 - 2 * delta) ** 3
                if map_0 + mar_0 > 2 * delta
                else 1e-10
            )

            num_good_categs = 0

            for ind, categ in enumerate(data):
                if ind == num_categs:
                    break

                tp, fp = 0, 0
                min_VP, max_VP, min_VR, max_VR = 1.0, 0.0, 1.0, 0.0

                # 1) find best d_f_macro
                best_d_f_macro = -1e-6
                for thr in range(1, len(categ.pred) + 2):
                    if thr <= len(categ.pred):
                        if categ.pred[thr - 1] == 1:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        tp = categ.size
                        fp = len(categ.pred) - categ.size
                    # for doc in categ.pred:
                    #     if doc == 1:
                    #         tp += 1
                    #     else:
                    #         fp += 1

                    p_k_thr: float = tp * 1.0 / (tp + fp)
                    r_k_thr: float = tp * 1.0 / categ.size
                    d_f_macro = alpha * p_k_thr + beta * r_k_thr

                    if d_f_macro > best_d_f_macro:
                        best_p_k = p_k_thr
                        best_r_k = r_k_thr
                        best_d_f_macro = d_f_macro

                # assert best_d_f_macro > -1e-6, str(categ.id)

                # 2) find estimated region - this cycle can be optimized
                # by storing
                tp, fp = 0, 0

                for thr in range(1, len(categ.pred) + 2):
                    if thr <= len(categ.pred):
                        if categ.pred[thr - 1] == 1:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        tp = categ.size
                        fp = len(categ.pred) - categ.size

                    p_k_thr = tp * 1.0 / (tp + fp)
                    r_k_thr = tp * 1.0 / categ.size
                    d_f_macro = alpha * p_k_thr + beta * r_k_thr

                    neighb = num_categs * eps * (map_0 + mar_0) ** 2
                    if d_f_macro > best_d_f_macro - neighb:
                        min_VP = min(min_VP, p_k_thr)
                        max_VP = max(max_VP, p_k_thr)

                        min_VR = min(min_VR, r_k_thr)
                        max_VR = max(max_VR, r_k_thr)

                if (map_0 > 0) or (mar_0 > 0):
                    res_p[0, i_p, i_r] += min_VP
                    res_p[1, i_p, i_r] += best_p_k
                    res_p[2, i_p, i_r] += max_VP

                    res_r[0, i_p, i_r] += min_VR
                    res_r[1, i_p, i_r] += best_r_k
                    res_r[2, i_p, i_r] += max_VR

                    # assert min_VP <= max_VP
                    # assert min_VR <= max_VR

                    num_good_categs += 1

            if num_good_categs == 0:
                continue

            res_p[:, i_p, i_r] /= num_good_categs
            res_r[:, i_p, i_r] /= num_good_categs
            # pbar.update()
    return res_p, res_r


# @jit(nopython=True)
def format_output(
    res_p: np.ndarray,
    res_r: np.ndarray,
    n_p: int,
    left_bottom: Tuple[float, float] = (0.0, 0.0),
    right_top: Tuple[float, float] = (1.0, 1.0),
) -> np.ndarray:
    """Format grid in expected format for further plotting."""
    p_0, r_0 = left_bottom
    p_1, r_1 = right_top
    n_r = n_p

    output = np.zeros(((n_p + 1) * (n_r + 1), 9))
    for i_p in range(n_p + 1):
        for i_r in range(n_r + 1):
            output[i_p * (n_p + 1) + i_r, 0] = p_0 + (p_1 - p_0) / n_p * i_p
            output[i_p * (n_p + 1) + i_r, 1] = r_0 + (r_1 - r_0) / n_r * i_r

    output[:, 2] = res_p[1].flatten()
    output[:, 3] = res_r[1].flatten()

    output[:, 4] = f1(output[:, 2], output[:, 3])

    output[:, 5] = res_p[0].flatten()  # min
    output[:, 6] = res_p[2].flatten()  # max

    output[:, 7] = res_r[0].flatten()  # min
    output[:, 8] = res_r[2].flatten()  # max

    return output
