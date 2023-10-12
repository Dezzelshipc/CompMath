# main element method
import numpy as np
import utility as gt


def get_max_index(mat: np.matrix, exclude: list):
    args = abs(mat).argmax(axis=1)
    maxes = abs(mat).max(axis=1)
    alist = [((k, args[k]), maxes[k]) for k in range(len(args)) if k not in exclude]
    argmax = max(alist, key=lambda x: x[1])
    return argmax[0]


def solve(matrix: np.matrix, values: np.array):
    matrix = matrix.copy().astype(float)
    values = values.copy().astype(float)

    if np.linalg.det(matrix) == 0:
        exit("Система несовместна!")

    rows_exclude = []
    for _ in range(len(matrix)):
        ind = get_max_index(matrix, rows_exclude)
        rows_exclude.append(ind[0])
        values[ind[0]] = values[ind[0]] / matrix[ind]
        matrix[ind[0]] = matrix[ind[0]] / matrix[ind]
        for i in range(len(matrix)):
            if i not in rows_exclude:
                values[i] -= matrix[(i, ind[1])] * values[ind[0]]
                matrix[i] -= matrix[(i, ind[1])] * matrix[ind[0]]

    rows_exclude.reverse()
    for i in rows_exclude:
        ind = matrix[i].argmax()
        for j in range(len(matrix)):
            if j != i:
                values[j] -= matrix[(j, ind)] * values[i]
                matrix[j] -= matrix[(j, ind)] * matrix[i]

    solution = np.array([0.] * len(matrix))
    for i in range(len(matrix)):
        ind = matrix[i].argmax()
        solution[ind] = values[i]

    return solution


if __name__ == "__main__":
    # gt.main_solve(solve, 10)
    gt.max_val_test(100, solve, 10)
