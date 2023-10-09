import numpy as np


def generate_matrix(size: int) -> np.matrix:
    return np.matrix(np.random.randint(-MAX_M, MAX_M, (size, size)))


def generate_values(size: int):
    return np.random.randint(-MAX_B, MAX_B, size)


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
    for _ in range(N):
        ind = get_max_index(matrix, rows_exclude)
        rows_exclude.append(ind[0])
        values[ind[0]] = values[ind[0]] / matrix[ind]
        matrix[ind[0]] = matrix[ind[0]] / matrix[ind]
        for i in range(N):
            if i not in rows_exclude:
                values[i] -= matrix[(i, ind[1])] * values[ind[0]]
                matrix[i] -= matrix[(i, ind[1])] * matrix[ind[0]]

    rows_exclude.reverse()
    for i in rows_exclude:
        ind = matrix[i].argmax()
        for j in range(N):
            if j != i:
                values[j] -= matrix[(j, ind)] * values[i]
                matrix[j] -= matrix[(j, ind)] * matrix[i]

    solution = np.array([0.] * N)
    for i in range(N):
        ind = matrix[i].argmax()
        solution[ind] = values[i]

    return solution


def main_solve():
    import time
    # A = np.matrix('1 2; 3 5')
    # b = np.array([1, 2])
    A = generate_matrix(N)
    b = generate_values(N)

    start_time = time.time()
    my_sol = solve(A, b)
    end_time = time.time() - start_time
    start_time_np = time.time()
    np_sol = np.linalg.solve(A, b)
    end_time_np = time.time() - start_time_np
    diff = my_sol - np_sol

    print(A, b, '', sep='\n')
    print(f"Обусловленность: {np.linalg.cond(A)}")

    print(f"{diff = }", f"{np.linalg.norm(diff) = }\n", f"{my_sol = }", f"{np_sol = }\n", sep='\n')
    print(*zip(my_sol, np_sol), sep='\n')
    print(end_time, end_time_np, end_time / end_time_np)


def test(iterations):
    differs = []
    for _ in range(iterations):
        A = generate_matrix(N)
        b = generate_values(N)

        my_sol = solve(A, b)
        np_sol = np.linalg.solve(A, b)
        diff = my_sol - np_sol
        differs.append(np.linalg.norm(diff, ord=1))
    print(f"Min: {min(differs)}", f"Max: {max(differs)}")


N = 10
MAX_M = 100
MAX_B = 100
np.random.seed(0)

if __name__ == "__main__":
    # main_solve()
    test(100)
