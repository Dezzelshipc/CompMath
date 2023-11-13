# square root method

import numpy as np
import utility as ut


def s_construct(mat: np.matrix):
    n = len(mat)
    mat = mat.astype(complex)
    s = np.zeros((n, n)).astype(complex)

    for i in range(n):
        s[i, i] = (mat[i, i] - sum(s[k, i] ** 2 for k in range(i))) ** 0.5
        for j in range(i + 1, n):
            s[i, j] = (mat[i, j] - sum(s[k, i] * s[k, j] for k in range(i))) / s[i, i]

    return s


def solve(matrix: np.matrix, values: np.array):
    matrix = matrix.copy().astype(complex)
    values = values.copy().astype(complex)

    n = len(matrix)

    s = s_construct(matrix)
    # print(np.round(s, 5))

    inner_sol = np.array([0.] * n).astype(complex)
    inner_sol[0] = values[0] / s[0, 0]
    for i in range(1, n):
        inner_sol[i] = (values[i] - sum(s[k, i] * inner_sol[k] for k in range(i))) / s[i, i]

    # print(np.round(inner_sol, 8))
    sol = np.array([0.] * n).astype(complex)

    sol[n - 1] = inner_sol[n - 1] / s[n - 1, n - 1]
    for i in reversed(range(n - 1)):
        sol[i] = (inner_sol[i] - sum(s[i, k] * sol[k] for k in range(i + 1, n))) / s[i, i]

    print(np.round(sol, 8))
    return sol.real


if __name__ == "__main__":
    # ut.main_solve(solve, 10)
    # ut.max_val_test(10000, solve, 10, plot=np.inf)
    A1, b1 = ut.read_data("ins.txt")
    ut.main_solve(solve, matrix=A1, values=b1)
    # s = s_construct(A1)
    # print(np.round(s, 5))
