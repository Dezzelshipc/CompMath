# invert method

import numpy as np
import utility as ut


def inverse(mat: np.matrix):
    n = len(mat)
    if n == 1:
        return np.matrix([1 / mat[0, 0]])

    mat_s_i = inverse(mat[:n - 1, :n - 1])
    print(mat_s_i)

    u = mat[:n - 1, n - 1]
    v = mat[n - 1, :n - 1]
    a_n = mat[n - 1, n - 1]

    va = v.dot(mat_s_i)
    au = mat_s_i.dot(u)

    alpha = a_n - va.dot(u)
    q = -va / alpha
    p = mat_s_i - au.dot(q)
    r = -au / alpha
    # print(p)

    mat_i = np.zeros((n, n))
    mat_i[:n - 1, :n - 1] = p
    mat_i[:n - 1, n - 1] = r.flatten()
    mat_i[n - 1, :n - 1] = q
    mat_i[n - 1, n - 1] = (1 / alpha)[0, 0]
    return mat_i


def solve(matrix: np.matrix, values: np.array):
    matrix = matrix.copy().astype(float)
    values = values.copy().astype(float)

    if np.linalg.det(matrix) == 0:
        exit("det = 0")

    inv = inverse(matrix)
    print(inv.dot(matrix))
    return inv.dot(values)


if __name__ == "__main__":
    # ut.max_val_test(1000, solve, 10, plot=np.inf)
    A1, b1 = ut.read_data("ins.txt")
    ut.main_solve(solve, matrix=A1, values=b1)
