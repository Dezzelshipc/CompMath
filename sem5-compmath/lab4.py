# lu method
import numpy as np
import utility as ut


def solve_single_div(matrix: np.matrix, values: np.array):
    matrix = matrix.copy().astype(float)
    values = values.copy().astype(float)
    eps = 1e-10

    if np.linalg.det(matrix) == 0:
        exit("Система несовместна!")

    for i in range(len(matrix)):
        if abs(matrix[i, i]) < eps:
            for k in range(i + 1, len(matrix)):
                if abs(matrix[k, i]) > eps:
                    matrix[i], matrix[k] = matrix[k].copy(), matrix[i].copy()
                    values[i], values[k] = values[k], values[i]
                    break

        for j in range(i + 1, len(matrix)):
            m = matrix[j, i] / matrix[i, i]
            values[j] -= m * values[i]
            matrix[j] -= m * matrix[i]

    # print(matrix, values)

    for i in reversed(range(len(matrix))):
        values[i] /= matrix[i, i]
        matrix[i] /= matrix[i, i]
        for j in range(i):
            values[j] -= matrix[j, i] * values[i]
            matrix[j] -= matrix[j, i] * matrix[i]

    # print(matrix, values)

    return values


def lu_decompose(matrix: np.matrix) -> (np.matrix, np.matrix):
    upper = np.zeros((len(matrix), len(matrix)))
    lower = np.identity(len(matrix))

    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i <= j:
                upper[i, j] = matrix[i, j] - sum(lower[i, k] * upper[k, j] for k in range(i))
            else:
                lower[i, j] = (matrix[i, j] - sum(lower[i, k] * upper[k, j] for k in range(j))) / upper[j, j]

    return lower, upper


def solve_exclusion(matrix: np.matrix, values: np.array):
    matrix = matrix.copy().astype(float)
    values = values.copy().astype(float)
    order = zip(np.count_nonzero(matrix, axis=1), range(len(matrix)))
    order = sorted(order, key=lambda x: x[0])
    order = [order[k][1] for k in range(len(order))]

    for k, i in enumerate(order):
        values[i] /= matrix[i, i]
        matrix[i] /= matrix[i, i]
        for j in order[k+1:]:
            values[j] -= matrix[j, i] * values[i]
            matrix[j] -= matrix[j, i] * matrix[i]

    return values


def solve_lu(matrix: np.matrix, values: np.array):
    if np.linalg.det(matrix) == 0:
        exit("Система несовместна!")
    # matrix = matrix.copy().astype(float)
    # values = values.copy().astype(float)
    lower, upper = lu_decompose(matrix)
    inner_sol = solve_exclusion(lower, values)  # Ly = b
    # print(inner_sol)
    return solve_exclusion(upper, inner_sol)    # Ux = y


if __name__ == "__main__":
    A = np.matrix('1 2 1 4;'
                  '2 0 4 3;'
                  '4 2 2 1;'
                  '-3 1 3 2')
    b = np.array([13, 28, 20, 6])
    # ut.main_solve(solve_lu, matrix=A, values=b)

    # ut.main_solve(solve_lu, 10)
    # ut.max_val_test(1000, solve, 10)
    A1, b1 = ut.read_data("in.txt")
    ut.main_solve(solve_lu, matrix=A1, values=b1)

