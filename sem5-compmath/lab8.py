# simple iteration method

import numpy as np
import utility as ut


def check_usability(matrix: np.matrix):
    n = len(matrix)
    for i in range(n):
        if abs(matrix[i, i]) <= sum(matrix[i, j] for j in range(n) if j != i) or abs(matrix[i, i]) <= sum(
                matrix[j, i] for j in range(n) if j != i):
            return False

    return True


def solve(matrix: np.matrix, values: np.array, eps: float = 1e-5):
    matrix = matrix.copy().astype(float)
    values = values.copy().astype(float)

    if not check_usability(matrix):
        exit("Matrix cannot be used in this method!")

    n = len(matrix)
    for i in range(n):
        values[i] /= matrix[i, i]
        matrix[i] /= -matrix[i, i]
        matrix[i, i] = 0

    values = np.matrix(values).reshape((n, 1))

    x = values.copy()
    iterations = 0
    while True:
        iterations += 1
        x_prev = x.copy()
        x = values + matrix.dot(x)
        if np.linalg.norm(x - x_prev) < eps:
            break

    print(f"{iterations = }")
    return np.array(x.flatten())


if __name__ == "__main__":
    # ut.main_solve(solve, 10)
    # ut.max_val_test(100, solve, 10, plot=np.inf)
    A1, b1 = ut.read_data("in.txt")
    ut.main_solve(solve, matrix=A1, values=b1, eps=1e-10)
