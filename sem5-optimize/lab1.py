import numpy as np

MAX_M = 3
MIN_M = -3
MAX_V = 3
MIN_V = -3


def generate_matrix(size: int, min_e=MIN_M, max_e=MAX_M) -> np.matrix:
    return np.matrix(np.random.randint(min_e, max_e, (size, size)))
    # return 2 * N * np.matrix(np.random.random_sample((size, size))) - N


def generate_values(size: int, min_e=MIN_V, max_e=MAX_V) -> np.array:
    return np.random.randint(min_e, max_e, (size, 1))


def xs(size: int):
    return np.ones((size, 1))


def generate_matrix_sim(size: int) -> np.matrix:
    mat = generate_matrix(size)
    # return (mat + mat.transpose()) / 2
    return np.matrix(np.dot(mat.transpose(), mat))


def f(var_vec: np.ndarray, mat: np.matrix, b_vec: np.ndarray) -> int | float:
    out = var_vec.transpose().dot(mat).dot(var_vec) / 2 + var_vec.transpose().dot(b_vec)
    return out.sum()


def f_grad(var_vec: np.ndarray, mat: np.matrix, b_vec: np.ndarray) -> np.ndarray:
    return mat.dot(var_vec) + b_vec


N = 3
np.random.seed(123)

A = generate_matrix_sim(N)
b = generate_values(N)
x = xs(N)
# print(np.linalg.det(A))
# A = np.matrix([[4, 0],
#                [0, 2]])
# x = np.array([[1],
#               [2]])
# b = np.array([[0],
#               [2]])
print(A, b, x, sep='\n')

eps = 1e-5
h = 1e-3
x_prev = x
x_prev2 = x + 2
f_prev = f(x_prev, A, b)


while np.linalg.norm(f_grad(x_prev, A, b)) > eps:
    # print(x_prev, h)
    x_new = x_prev - h * f_grad(x_prev, A, b)
    f_new = f(x_new, A, b)
    if np.linalg.norm(f_grad(x_new, A, b)) < eps or h < 1e-100:
        x_prev = x_new
        break

    if f_new > f_prev or np.linalg.norm(x_prev2 - x_new) < eps and f_prev == f_new:
        h /= 2
        continue
    x_prev2 = x_prev
    x_prev = x_new
    f_prev = f_new

print(x_prev, f_prev, sep='\n')
