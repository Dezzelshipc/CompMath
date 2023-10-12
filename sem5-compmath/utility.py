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


def main_solve(solve_func, size=5, matrix=None, values=None, is_time=False):
    import time
    if matrix is None:
        matrix = generate_matrix(size)
    if values is None:
        values = generate_values(size)

    start_time = time.time()
    my_sol = solve_func(matrix, values)
    end_time = time.time() - start_time
    start_time_np = time.time()
    np_sol = np.linalg.solve(matrix, values)
    end_time_np = time.time() - start_time_np
    diff = my_sol - np_sol

    print(matrix, values, '', sep='\n')
    print(f"Обусловленность: {np.linalg.cond(matrix)}")

    print(f"{diff = }", f"{np.linalg.norm(diff) = }\n", f"{my_sol = }", f"{np_sol = }\n", sep='\n')
    print(*zip(my_sol, np_sol), sep='\n')
    if is_time:
        print(end_time, end_time_np, end_time / end_time_np if end_time_np > 0 else "-")


def max_val_test(iterations, solve_func, size=5, norm_ord=1):
    differs = []
    for _ in range(iterations):
        A = generate_matrix(size)
        b = generate_values(size)

        my_sol = solve_func(A, b)
        np_sol = np.linalg.solve(A, b)
        diff = my_sol - np_sol
        differs.append(np.linalg.norm(diff, ord=norm_ord))
    print(f"Min: {min(differs)}", f"Max: {max(differs)}")


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
