# richardson method

import numpy as np
import utility as ut
import math


def solve(matrix: np.matrix, values: np.array, eps: float = 1e-5) -> (np.array, int):
    matrix = matrix.copy().astype(float)
    values = values.copy().astype(float)

    eigen = np.linalg.eig(matrix)

    eigen_max = eigen.eigenvalues.max()
    eigen_min = eigen.eigenvalues.min()

    eta = eigen_min / eigen_max
    ro0 = (1 - eta) / (1 + eta)
    ro1 = (1 - math.sqrt(eta)) / (1 + math.sqrt(eta))
    n = math.log(2 / eps) / math.log(1 / ro1)
    # print(n, math.log(eps / 2, ro1))
    t0 = 2 / (eigen_min + eigen_max)

    values = np.matrix(values).T

    x = np.zeros((len(matrix), 1))
    iterations = 0
    for k in range(math.ceil(n) + 1):
        iterations += 1
        d = math.cos((2 * k + 1) * math.pi / 2 * n)
        t = t0 / (1 + ro0 * d)
        x = t * (values - matrix.dot(x)) + x

    return np.array(x.flatten()), iterations


if __name__ == "__main__":
    A1, b1 = ut.read_data("in_i.txt")
    ut.iter_solve(solve, matrix=A1, values=b1, eps=1e-10)
