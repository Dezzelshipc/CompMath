# method optimal exclusion
import numpy as np
import utility as ut


def solve(matrix: np.matrix, values: np.array):
    matrix = matrix.copy().astype(float)
    values = values.copy().astype(float)
    order = list(range(len(matrix)))

    for i in range(len(matrix)):
        for j in range(i):
            m = matrix[i, j]
            values[i] -= m * values[j]
            matrix[i] -= m * matrix[j]

        max_j = abs(matrix[i, i:]).argmax() + i
        matrix[:, [i, max_j]] = matrix[:, [max_j, i]]
        order[i], order[max_j] = order[max_j], order[i]

        values[i] /= matrix[i, i]
        matrix[i] /= matrix[i, i]

        for j in range(i):
            m = matrix[j, i]
            values[j] -= m * values[i]
            matrix[j] -= m * matrix[i]

        # print(matrix)

    solution = np.array([0.] * len(matrix))
    for i, o in enumerate(order):
        solution[o] = values[i]

    return solution


if __name__ == "__main__":
    A = np.matrix('1 2 1 4;'
                  '2 0 4 3;'
                  '4 2 2 1;'
                  '-3 1 3 2')
    b = np.array([13, 28, 20, 6])
    # ut.main_solve(solve, matrix=A, values=b)

    # ut.main_solve(solve, 10)
    # ut.max_val_test(1000, solve, 10)
    A1, b1 = ut.read_data("in2.txt")
    ut.main_solve(solve, matrix=A1, values=b1)
