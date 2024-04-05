# mirror method

import numpy as np
import utility as ut


def sigma(num: float):
    return 1 if num >= 0 else -1


def solve(matrix: np.matrix, values: np.array):
    matrix = matrix.copy().astype(float)
    values = values.copy().astype(float)

    n = len(matrix)

    for k in range(n):
        p = np.zeros(n)
        p[k] = matrix[k, k] + sigma(matrix[k, k]) * (sum(matrix[l, k] ** 2 for l in range(k, n))) ** 0.5
        p[k + 1:] = matrix[k + 1:, k].flatten()

        matrix[k, k] = - sigma(matrix[k, k]) * (sum(matrix[l, k] ** 2 for l in range(k, n))) ** 0.5
        values_new = values.copy()
        matrix_new = matrix.copy()
        for i in range(k, n):
            values_new[i] = values[i] - 2 * p[i] * sum(p[l] * values[l] for l in range(k, n)) / sum(
                p[l] ** 2 for l in range(k, n))
            for j in range(k+1, n):
                matrix_new[i, j] = matrix[i, j] - 2 * p[i] * sum(p[l] * matrix[l, j] for l in range(k, n)) / sum(
                    p[l] ** 2 for l in range(k, n))
        matrix_new[k + 1:, k] = np.zeros((n - k - 1, 1))

        matrix = matrix_new
        values = values_new
        # print(matrix)

    # print(values)
    sol = np.array([0.] * n).astype(float)

    for i in reversed(range(n)):
        sol[i] = (values[i] - sum(matrix[i, k] * sol[k] for k in range(i + 1, n))) / matrix[i, i]

    return sol


if __name__ == "__main__":
    # ut.main_solve(solve, 10)
    # ut.max_val_test(100, solve, 10, plot=np.inf)
    A1, b1 = ut.read_data("sem5-compmath/in_i.txt")
    ut.main_solve(solve, matrix=A1, values=b1)
