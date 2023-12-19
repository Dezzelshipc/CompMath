# rotation method

import numpy as np
import utility as ut
import math


def sign(num):
    return 1 if num >= 0 else -1


def solve(matrix: np.matrix, p: int = 3) -> (np.array, int):
    matrix = matrix.copy().astype(float)

    n = len(matrix)

    p_i = 0
    sigma = np.max(abs(matrix)) + 1

    iterations = 0
    matrix_vectors = np.eye(n)

    stop_flag = False
    while True:
        diag = np.diagflat(matrix.diagonal())
        mat_wo_diag = abs(matrix - diag) - np.eye(n)
        flat_index = mat_wo_diag.argmax()
        i, j = np.unravel_index(flat_index, diag.shape)

        while abs(matrix[i, j]) < sigma:
            p_i += 1
            if p_i > p:
                stop_flag = True
                break
            sigma = math.sqrt(np.max(abs(diag))) * 10 ** (-p_i)
        if stop_flag:
            break
        iterations += 1

        aii = matrix[i, i]
        ajj = matrix[j, j]
        aij = matrix[i, j]
        d = math.sqrt((aii - ajj) ** 2 + 4 * aij ** 2)
        c = math.sqrt((1 + abs(aii - ajj) / d) / 2)
        s = sign(aij * (aii - ajj)) * math.sqrt((1 - abs(aii - ajj) / d) / 2)

        rot = np.eye(n)
        rot[i, i] = c
        rot[j, j] = c
        rot[i, j] = -s
        rot[j, i] = s
        matrix_vectors = matrix_vectors.dot(rot)

        new_matrix = matrix.copy()

        for k in range(n):
            new_matrix[k, i] = new_matrix[i, k] = c * matrix[k, i] + s * matrix[k, j]
            new_matrix[k, j] = new_matrix[j, k] = -s * matrix[k, i] + c * matrix[k, j]

        new_matrix[i, i] = c ** 2 * aii + 2 * c * s * aij + s ** 2 * ajj
        new_matrix[j, j] = s ** 2 * aii - 2 * c * s * aij + c ** 2 * ajj
        new_matrix[i, j] = new_matrix[j, i] = 0

        matrix = new_matrix

    for i in range(len(matrix_vectors)):
        matrix_vectors[i] /= np.linalg.norm(matrix_vectors[i])

    return matrix_vectors.T, np.array(matrix.diagonal())[0], iterations


if __name__ == "__main__":
    A1, b1 = ut.read_data("in_e.txt")
    ut.eigen_full_solve(solve, matrix=A1, p =7)
