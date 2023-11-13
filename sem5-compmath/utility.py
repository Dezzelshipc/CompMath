import math
import matplotlib.pyplot as plt

import numpy as np

MAX_M = 10
MIN_M = -10
MAX_V = 10
MIN_V = -10


def generate_matrix(size: int, min_e=MIN_M, max_e=MAX_M) -> np.matrix:
    return np.matrix(np.random.randint(min_e, max_e, (size, size)))
    # return 2 * N * np.matrix(np.random.random_sample((size, size))) - N


def generate_values(size: int, min_e=MIN_V, max_e=MAX_V) -> np.array:
    return np.random.randint(min_e, max_e, size)


def main_solve(solve_func, size=5, matrix=None, values=None, eps=None, is_time=False):
    import time
    if matrix is None:
        matrix = generate_matrix(size)
    if values is None:
        values = generate_values(size)

    start_time = time.time()
    if type(eps) is float or type(eps) is int:
        my_sol = solve_func(matrix, values, eps)
    else:
        my_sol = solve_func(matrix, values)
    end_time = time.time() - start_time
    start_time_np = time.time()
    np_sol = np.linalg.solve(matrix, values)
    end_time_np = time.time() - start_time_np
    diff = my_sol - np_sol

    print(matrix, values, '', sep='\n')
    print(f"Обусловленность: {np.linalg.cond(matrix)}")

    print(f"{my_sol = }", f"{np_sol = }\n", f"{diff = }", f"{max(abs(diff)) = }", f"{np.linalg.norm(diff) = }",
          sep='\n')
    # print(*zip(my_sol, np_sol), sep='\n')
    if is_time:
        print(end_time, end_time_np, end_time / end_time_np if end_time_np > 0 else "-")


def max_val_test(iterations, solve_func, size=5, norm_ord=1, plot=False):
    minmax = [1e100, -1e100]
    counts = dict()
    for _ in range(iterations):
        A = generate_matrix(size)
        b = generate_values(size)

        my_sol = solve_func(A, b)
        np_sol = np.linalg.solve(A, b)
        diff = np.linalg.norm(my_sol - np_sol, ord=norm_ord)
        minmax = [min(minmax[0], diff), max(minmax[1], diff)]

        power = round(math.log10(diff), 1)
        if power not in counts:
            counts[power] = 0
        counts[power] += 1
        if power > -10:
            print(power, np.linalg.cond(A))

    if plot:
        print(counts)
        order = sorted(counts, key=lambda x: x)
        values = [counts[x] for x in order]

        plt.bar(order, values, edgecolor='k', align='edge', width=0.1)
        plt.show()
    print(f"Min: {min(minmax)}", f"Max: {max(minmax)}")


def solve_exclusion(matrix: np.matrix, values: np.array):
    matrix = matrix.copy().astype(float)
    values = values.copy().astype(float)
    order = zip(np.count_nonzero(matrix, axis=1), range(len(matrix)))
    order = sorted(order, key=lambda x: x[0])
    order = [order[k][1] for k in range(len(order))]

    for k, i in enumerate(order):
        values[i] /= matrix[i, i]
        matrix[i] /= matrix[i, i]
        for j in order[k + 1:]:
            values[j] -= matrix[j, i] * values[i]
            matrix[j] -= matrix[j, i] * matrix[i]

    return values


def matrix_str(matrix: np.matrix, values: np.array):
    s = ""
    for i in range(len(matrix)):
        s += f"{[matrix[i, j] for j in range(len(matrix))]} {values[i]}\n"
    return s


def read_data(file_name: str) -> (np.matrix, np.ndarray):
    with open(file_name, "r") as f:
        start = f.readline().split()
        raw_mat = [start]
        for _ in range(len(start) - 1):
            raw_mat.append(f.readline().split())

        s = ''
        while s.strip() == '':
            s = f.readline()

        return np.matrix(raw_mat).astype(float), np.array(s.split()).astype(float)


if __name__ == "__main__":
    print(read_data("in.txt"))
