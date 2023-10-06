import math

import numpy as np

N = 2
np.random.seed(1)


def generate_matrix(size: int):
    return np.random.randint(1, 3, (size, size))


def generate_matrix_sim(size: int) -> np.matrix:
    mat = generate_matrix(size)
    # return (mat + mat.transpose()) / 2
    return np.matrix(np.dot(mat.transpose(), mat))


def f(var_vec: np.ndarray, mat: np.matrix, b_vec: np.ndarray) -> int | float:
    out = var_vec.transpose().dot(mat).dot(var_vec) / 2 + var_vec.transpose().dot(b_vec)
    return out.sum()


def f_grad(var_vec: np.ndarray, mat: np.matrix, b_vec: np.ndarray) -> np.ndarray:
    return mat.dot(var_vec) + b_vec


def norm(var_vec: np.ndarray):
    return math.sqrt(var_vec.transpose().dot(var_vec))


# A = generate_matrix_sim(N)
A = np.matrix([[4, 0],
               [0, 2]])
x = np.array([[1],
              [2]])
b = np.array([[0],
              [2]])

eps = 1e-4
h = 1
x_prev = x
x_prev2 = x * 2
f_prev = f(x_prev, A, b)

while True:
    x_new = x_prev - h * f_grad(x_prev, A, b)
    f_new = f(x_new, A, b)
    if f_prev < f_new or norm(x_prev2 - x_new) < eps:
        h = h / 2
        continue

    print(f_prev, f_new, x_new, x_prev, h)
    if abs(f_prev - f_new) < eps and norm(x_new - x_prev) < eps:
        f_prev = f_new
        x_prev = x_new
        break
    x_prev2 = x_prev
    x_prev = x_new
    f_prev = f_new

print(f_prev, x_prev)
