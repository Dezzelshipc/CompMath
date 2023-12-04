# fastest gradient descent method

import numpy as np
import utility as ut


def solve(matrix: np.matrix, values: np.array, eps: float = 1e-5, iter_max=1e5):
    x = np.matrix(values).T
    values = np.matrix(values).T
    iterations = 0
    while True:
        iterations += 1
        x_prev = x.copy()
        r = matrix.dot(x) - values
        wwTr = matrix.dot(matrix.T).dot(r)
        m = r.T.dot(wwTr) / wwTr.T.dot(wwTr)
        x = x - m[0, 0] * matrix.T.dot(r)
        if np.linalg.norm(x - x_prev) < eps or iterations > iter_max:
            break

    return np.array(x.flatten()), iterations


if __name__ == "__main__":
    A1, b1 = ut.read_data("in_i.txt")
    ut.iter_solve(solve, matrix=A1, values=b1, eps=1e-10)
