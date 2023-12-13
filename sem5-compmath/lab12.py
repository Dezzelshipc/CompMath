# rotation method

import numpy as np
import utility as ut
import math


def sign(num):
    return 1 if num >= 0 else -1


def solve(matrix: np.matrix, sigmas=None, p: int = 3) -> (np.array, int):
    matrix = matrix.copy().astype(float)

    n = len(matrix)

    if not sigmas:
        sigmas = (10 ** (-k) for k in range(1, p + 1))

    sigma = next(sigmas)
    iterations = 0
    matrix_vectors = np.eye(n)
    while True:
        diag = np.diagflat(matrix.diagonal())
        flat_index = abs(matrix - diag).argmax()
        i, j = np.unravel_index(flat_index, diag.shape)
        if abs(matrix[i, j]) < sigma:
            try:
                sigma = next(sigmas)
            except StopIteration:
                break
        iterations += 1

        aii = matrix[i, i]
        ajj = matrix[j, j]
        aij = matrix[i, j]
        d = math.sqrt((aii - ajj) ** 2 + 4 * aij ** 2)
        c = math.sqrt((1 + abs(aii - ajj) / d) / 2)
        s = sign(aij * (aii - ajj)) * math.sqrt((1 - abs(aii - ajj) / d) / 2)

        T = np.eye(n)
        T[i, i] = T[j, j] = c
        T[i, j] = -s
        T[j, i] = s
        matrix_vectors = matrix_vectors.dot(T)

        new_matrix = matrix.copy()

        for k in range(n):
            new_matrix[k, i] = new_matrix[i, k] = c * matrix[k, i] + s * matrix[k, j]
            new_matrix[k, j] = new_matrix[j, k] = -s * matrix[k, i] + c * matrix[k, j]

        new_matrix[i, i] = c ** 2 * aii + 2 * c * s * aij + s ** 2 * ajj
        new_matrix[j, j] = s ** 2 * aii - 2 * c * s * aij + c ** 2 * ajj
        new_matrix[i, j] = new_matrix[j, i] = 0

        matrix = new_matrix

    print(matrix)

    return matrix_vectors, matrix.diagonal(), iterations


if __name__ == "__main__":
    A1, b1 = ut.read_data("in_e4.txt")
    ut.eigen_full_solve(solve, matrix=A1)
