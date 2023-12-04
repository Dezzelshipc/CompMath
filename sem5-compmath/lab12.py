# eigenvalues simple iteration

import numpy as np
import utility as ut


def max_eigen(matrix: np.matrix, eps: float = 1e-5, iter_max=1e5):
    x = np.ones((len(matrix), 1))
    eig_val = 0
    iterations = 0
    while True:
        iterations += 1
        x_prev = x.copy()
        y = matrix.dot(x)
        eig_val = y.T.dot(x_prev)[0, 0]
        x = y / np.linalg.norm(y)
        if np.linalg.norm(np.sign(eig_val) * x - x_prev) < eps or iterations > iter_max:
            break

    x = np.array(x.flatten())[0]
    return x, eig_val, iterations


if __name__ == "__main__":
    A1, b1 = ut.read_data("in_e.txt")
    ut.eigen_solve(max_eigen, matrix=A1, eps=1e-10)
