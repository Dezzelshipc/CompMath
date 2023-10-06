import numpy as np


def generate_matrix(size: int) -> np.matrix:
    return np.matrix(np.random.randint(-MAX_M, MAX_M, (size, size)))


def generate_values(size: int):
    return np.random.randint(-MAX_B, MAX_B, size)


def get_max_index(mat: np.matrix, exclude: list):
    args = list(zip(abs(mat).argmax(axis=1), abs(mat).max(axis=1)))
    args = [((k, v[0][0].max()), v[1][0].max()) for k, v in enumerate(args) if k not in exclude]
    argmax = max(args, key=lambda x: x[1])
    return argmax[0]


def solve(matrix: np.matrix, values: np.array):
    matrix = matrix.copy().astype(float)
    values = values.copy().astype(float)

    if np.linalg.det(matrix) == 0:
        exit("Система несовместна!")

    rows_exclude = []
    for _ in range(N):
        ind = get_max_index(matrix, rows_exclude)
        # print(matrix, values, ind, '', sep='\n')
        rows_exclude.append(ind[0])
        values[ind[0]] = values[ind[0]] / matrix[ind]
        matrix[ind[0]] = matrix[ind[0]] / matrix[ind]
        for i in range(N):
            if i not in rows_exclude:
                values[i] -= matrix[(i, ind[1])] * values[ind[0]]
                matrix[i] -= matrix[(i, ind[1])] * matrix[ind[0]]

    rows_exclude.reverse()
    rows_back_exclude = []
    for i in rows_exclude:
        ind = matrix[i].argmax()
        rows_back_exclude.append(i)
        for j in range(N):
            if j not in rows_back_exclude:
                values[j] -= matrix[(j, ind)] * values[i]
                matrix[j] -= matrix[(j, ind)] * matrix[i]

    solution = np.array([0.] * N)
    for i in range(N):
        ind = matrix[i].argmax()
        solution[ind] = values[i]

    return solution


N = 5
MAX_M = 10
MAX_B = 10
np.random.seed()

# A = np.matrix('1 2; 3 5')
# b = np.array([1, 2])
A = generate_matrix(N)
b = generate_values(N)

my_sol = solve(A, b)
np_sol = np.linalg.solve(A, b)

print(A, b, '', sep='\n')
print(f"Обусловленность: {np.linalg.cond(A)}")

print(f"{my_sol-np_sol = }", f"{my_sol = }", f"{np_sol = }", sep='\n')
