# eigenvalues simple iteration

import numpy as np
import utility as ut
import lab4


def max_eigen(matrix: np.matrix, eps: float = 1e-5, iter_max=1e5):
    lower, upper = lab4.lu_decompose(matrix)

    x = np.ones((len(matrix), 1))
    eig_val = 1 / abs(x).max()
    iterations = 0
    while True:
        iterations += 1
        inner_sol = lab4.solve_exclusion(lower, x * eig_val)  # Ly = b
        x = lab4.solve_exclusion(upper, inner_sol)
        # print(x.T)
        eig_val_prev = eig_val
        eig_val = 1 / abs(x).max()
        if abs(eig_val - eig_val_prev) < eps or iterations > iter_max:
            break

    x = np.array(x.flatten()) / np.linalg.norm(x)
    return x, eig_val, iterations


if __name__ == "__main__":
    A1, b1 = ut.read_data("in_e.txt")
    ut.eigen_solve(max_eigen, matrix=A1, eps=1e-10)
