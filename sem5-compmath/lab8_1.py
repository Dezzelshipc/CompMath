# relaxation method

import numpy as np
import utility as ut


def solve(matrix: np.matrix, values: np.array, eps: float = 1e-5, w: float = 1, iter_max=1e5):
    if w <= 0 or w >= 2:
        exit(f"Invalid value of {w = }. w must be in (0, 2)")
    x = values.copy()
    iterations = 0
    while True:
        # print(x)
        iterations += 1
        x_prev = x.copy()
        for i in range(len(matrix)):
            x[i] = (1 - w) * x[i] + w / matrix[i, i] * (values[i] - sum(matrix[i, j] * x[j] for j in range(i)) - sum(
                matrix[i, j] * x[j] for j in range(i + 1, len(matrix))))
        if np.linalg.norm(x - x_prev, ord=1) < eps or iterations > iter_max:
            break

    return x, iterations


if __name__ == "__main__":
    A1, b1 = ut.read_data("in_i.txt")
    print(A1, b1, sep='\n')
    print(np.linalg.solve(A1, b1))
    for w in [1e-3, 0.5, 1, 1.5, 2 - 1e-3]:
        print(f"{w = }:", solve(A1, b1, eps=1e-10, w=w))

    # ut.iter_solve(solve, matrix=A1, values=b1, eps=1e-10, w=0.5)
