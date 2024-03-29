from typing import Tuple
from numba import jit
from numba.typed import List
import numpy as np

from .convex_hull import get_convex_hull


@jit(nopython=True)
def estimate_grid(
    data: List, num_categs: int, n_p: int, num_docs: int = -1, beta: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Estimating F-macro in every point of a unit square grid,
    the number of points in grid is (n_p + 1) * (n_p + 1).
    """
    num_docs = len(data[0].pred) if num_docs == -1 else num_docs
    num_categs = len(data) if len(data) < num_categs else num_categs
    print(f"num categories = {num_categs}")

    n_r = n_p  # the same amount of points in x and y axis
    p_0, p_1, r_0, r_1 = 0.0, 1.0, 0.0, 1.0

    res_p = np.zeros((3, n_p + 1, n_r + 1), dtype=np.float64)
    res_r = np.zeros((3, n_p + 1, n_r + 1), dtype=np.float64)

    for i_p in range(n_p + 1):
        for i_r in range(n_r + 1):

            map_0: float = p_0 + (p_1 - p_0) / n_p * i_p
            mar_0: float = r_0 + (r_1 - r_0) / n_r * i_r

            w_1, w_2 = mar_0 ** 2, beta ** 2 * map_0 ** 2
            delta = 1.0 / num_categs

            eps = (
                2
                * delta ** 2
                * (map_0 + mar_0 + 2 * delta) ** 2
                / (map_0 + mar_0 - 2 * delta) ** 3
                if map_0 + mar_0 > 2 * delta
                else 1e10
            )

            num_good_categs = 0

            for ind in range(num_categs):
                categ = data[ind]

                tp, fp = 0, 0
                min_VP, max_VP, min_VR, max_VR = 1.0, 0.0, 1.0, 0.0

                # 1) find best d_f_macro
                best_d_f_macro = -1e6
                best_p_k: float = 0.0
                for thr in range(1, len(categ.pred) + 2):
                    if thr <= len(categ.pred):
                        if categ.pred[thr - 1] == 1:
                            tp += 1
                        else:
                            fp += 1
                    else:
                        tp = categ.size
                        fp = num_docs - categ.size

                    p_k_thr: float = tp * 1.0 / (tp + fp)
                    r_k_thr: float = tp * 1.0 / categ.size
                    d_f_macro = w_1 * p_k_thr + w_2 * r_k_thr

                    if d_f_macro > best_d_f_macro + 1e-12:
                        best_p_k = p_k_thr
                        best_r_k = r_k_thr
                        best_d_f_macro = d_f_macro

                assert best_d_f_macro > -1e6

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
                        fp = num_docs - categ.size

                    p_k_thr = tp * 1.0 / (tp + fp)
                    r_k_thr = tp * 1.0 / categ.size
                    d_f_macro = w_1 * p_k_thr + w_2 * r_k_thr

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

                    assert min_VP <= max_VP
                    assert min_VR <= max_VR

                    num_good_categs += 1

            if num_good_categs == 0:
                continue

            res_p[:, i_p, i_r] /= num_good_categs
            res_r[:, i_p, i_r] /= num_good_categs

    return res_p, res_r


@jit(nopython=True)
def estimate_DV(
    data: List, num_categs: int, num_angles: int, num_docs: int = -1
) -> np.ndarray:
    """Returns coordinates of D(V) border"""
    d_angle = 2.0 * np.pi / num_angles
    num_docs = len(data[0].pred) if num_docs == -1 else num_docs
    num_categs = len(data) if len(data) < num_categs else num_categs
    i0 = 0
    hull = np.zeros((num_angles, 2))
    count = 0  # initial weight of start hull
    for angle_i in range(num_angles):
        hull[angle_i] = [0.2, 0.2]
    for i in range(num_categs):
        cat = data[i0 + i]
        catPR = np.zeros((cat.pred.size + 1, 2))
        tp, fp = 0, 0
        for barrier in range(1, cat.pred.size + 2):
            if barrier <= cat.pred.size:
                if cat.pred[barrier - 1] == 1:
                    tp += 1
                else:
                    fp += 1
            else:
                tp = cat.size
                fp = num_docs - cat.size
            catPR[barrier - 1, 0] = tp * 1.0 / (tp + fp)
            catPR[barrier - 1, 1] = tp * 1.0 / cat.size
        # ind = scipy.spatial.ConvexHull(catPR).vertices
        # catPR = catPR[ind, :]
        catPR = get_convex_hull(catPR)
        assert len(catPR) > 1
        for pi in range(len(catPR)):
            prev = catPR[len(catPR) - 1] if pi == 0 else catPR[pi - 1]
            cur = catPR[pi]
            next = catPR[0] if pi == len(catPR) - 1 else catPR[pi + 1]
            a = complex(prev[0] - cur[0], prev[1] - cur[1])
            b = complex(next[0] - cur[0], next[1] - cur[1])
            phi1 = (np.angle(a) + np.pi / 2) % (2 * np.pi)
            phi2 = (np.angle(b) - np.pi / 2) % (2 * np.pi)
            if phi1 < phi2:
                for angle_i in range(
                    int(np.ceil(phi1 / d_angle)), int(phi2 // d_angle) + 1
                ):
                    hull[angle_i, 0] = (hull[angle_i, 0] * count + cur[0]) / (
                        count + 1
                    )
                    hull[angle_i, 1] = (hull[angle_i, 1] * count + cur[1]) / (
                        count + 1
                    )
            else:  # we jump over 0
                for angle_i in range(0, int(phi2 // d_angle) + 1):
                    hull[angle_i, 0] = (hull[angle_i, 0] * count + cur[0]) / (
                        count + 1
                    )
                    hull[angle_i, 1] = (hull[angle_i, 1] * count + cur[1]) / (
                        count + 1
                    )
                for angle_i in range(int(np.ceil(phi1 / d_angle)), num_angles):
                    hull[angle_i, 0] = (hull[angle_i, 0] * count + cur[0]) / (
                        count + 1
                    )
                    hull[angle_i, 1] = (hull[angle_i, 1] * count + cur[1]) / (
                        count + 1
                    )
        count += 1
    return hull
