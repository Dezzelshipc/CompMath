# relaxation method

import numpy as np
import utility as ut


def solve(matrix: np.matrix, values: np.array, eps: float = 1e-5):
    matrix = matrix.copy().astype(float)
    values = values.copy().astype(float)

    n = len(matrix)
    for i in range(n):
        values[i] /= matrix[i, i]
        matrix[i] /= -matrix[i, i]
        matrix[i, i] = 0

    x = np.zeros(n)
    nev = values.copy()

    iterations = 0
    while np.linalg.norm(nev) >= eps:
        iterations += 1
        k = abs(nev).argmax()
        x[k] += nev[k]
        nev = np.array([nev[i] + matrix[i, k] * nev[k] if i != k else 0 for i in range(n)])

    # print(x, nev)
    print(f"{iterations = }")
    return x


if __name__ == "__main__":
    # ut.main_solve(solve, 10)
    # ut.max_val_test(100, solve, 10, plot=np.inf)
    A1, b1 = ut.read_data("in.txt")
    ut.main_solve(solve, matrix=A1, values=b1, eps=1e-10)
