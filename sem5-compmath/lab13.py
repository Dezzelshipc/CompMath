# eigenvalues simple iteration

import numpy as np
import utility as ut
import lab4


def max_eigen(matrix: np.matrix, eps: float = 1e-5):
    lower, upper = lab4.lu_decompose(matrix)

    x = np.ones((len(matrix), 1))
    eig_val = 0
    iterations = 0
    while True:
        iterations += 1
        x_prev = x.copy()
        inner_sol = lab4.solve_exclusion(lower, x / abs(x).max())  # Ly = b
        x = lab4.solve_exclusion(upper, inner_sol)
        if abs(1 / abs(x).max() - 1 / abs(x_prev).max()) < eps:
            eig_val = 1 / abs(x).max()
            break

    x = np.array(x.flatten()) / np.linalg.norm(x)
    return x, eig_val, iterations


if __name__ == "__main__":
    A1, b1 = ut.read_data("in_e.txt")
    ut.eigen_solve(max_eigen, matrix=A1, eps=1e-10)
