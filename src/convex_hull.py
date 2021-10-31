from numba import jit
import numpy as np


@jit(nopython=True)
def cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


@jit(nopython=True)
def get_convex_hull(p: np.array) -> np.array:
    n = p.shape[0]
    k = 0
    if n == 1:
        return p
    hull = np.zeros((n * 2, 2))

    # sort points lexicographically
    x1 = np.sort(p[:, 0])
    dx1 = x1[1:] - x1[:-1]
    mdx1 = np.min(dx1[dx1 > 0])
    mnoj = 2 / mdx1
    ind = np.argsort(p[:, 0] * mnoj + p[:, 1])
    p = p[ind]

    # build lower hull
    for ind, point in enumerate(p):
        while k >= 2 and cross(hull[k - 2], hull[k - 1], point) <= 0:
            k -= 1
        hull[k] = point
        k += 1

    # build upper hull
    t = k + 1
    for ind in range(n - 2, -1, -1):
        while k >= t and cross(hull[k - 2], hull[k - 1], p[ind]) <= 0:
            k -= 1
        hull[k] = p[ind]
        k += 1

    # remove trailing zeros
    hull = hull[: (k - 1), :]
    return hull