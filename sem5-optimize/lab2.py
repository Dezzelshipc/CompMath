import numpy as np


def f(var_vec: np.ndarray, mat: np.matrix, b_vec: np.ndarray, center: np.ndarray, radius: float) -> int | float:
    n = len(mat)
    vv = var_vec[:n]
    out = vv.T.dot(mat).dot(vv) / 2 + vv.T.dot(b_vec) + var_vec[n] * (
            np.linalg.norm(vv[:n] - center) ** 2 - radius ** 2)
    return out.sum()


def f_grad(var_vec: np.ndarray, mat: np.matrix, b_vec: np.ndarray, center: np.ndarray) -> np.ndarray:
    n = len(mat)
    vv = var_vec[:n, 0]
    return mat.dot(vv) + b_vec + 2 * var_vec[n, 0] * (vv - center)


def g_func(var_vec: np.ndarray, center: np.ndarray, radius: float):
    return np.linalg.norm(var_vec[:-1] - center) ** 2 - radius ** 2


def w_mat(vv: np.ndarray, mat: np.matrix, center: np.ndarray) -> np.matrix:
    n = len(mat)
    W = np.zeros((n + 1, n + 1))
    W[:n, :n] = mat + 2 * vv[n, 0] * np.eye(n)

    W[n, :n] = (2 * (vv[:n, 0] - center)).flatten()
    W[:n, n] = (2 * (vv[:n, 0] - center)).flatten()
    W[n, n] = 0
    return W


def g_vec(vv: np.ndarray, mat: np.matrix, b_vec: np.ndarray, center: np.ndarray, radius: float) -> np.matrix:
    n = len(mat)
    G = np.zeros((len(vv), 1))
    G[:n] = f_grad(vv, mat, b_vec, center)
    G[n] = (np.linalg.norm(vv[:n, 0] - center) ** 2 - radius ** 2)
    return G


# A = np.matrix([[1, 3, -2, 0],
#                [3, 4, -5, 1],
#                [-2, -5, 3, -2],
#                [0, 1, -2, 5]])
# b = np.matrix([[1], [1], [1], [1]])
# c = np.matrix([[1], [1], [1], [1]])
# r = 2
# l = 1

# A = np.matrix([[-8, 2, 2, 3],
#                [2, -1, 1, 5],
#                [2, 1, 4, 3],
#                [3, 5, 3, 6]])

# b = np.matrix([[2], [6], [5], [3]])
# c = np.matrix([[5], [5], [5], [2]])
# r = 2
# l = 1

A = np.matrix([[8., 3., 6., 6.],
               [3., 4., 5., 5.],
               [6., 5., 4., 7.],
               [6., 5., 7., 6.]])
b = np.matrix([2., 6., 5., 3.]).T
x0 = np.matrix([2., 3., 0., 0.]).T
l = 1

c = np.matrix([5., 5., 5., 2.]).T
r = 1

for i in range(len(A)):
    print(np.linalg.det(A[:i + 1, :i + 1]), end=' ')
print()

mins = []

n = len(A)
x = np.matrix([[0.]] * (n + 1))

eps = 1e-10

for i in range(n * 5):
    k = 0
    x[:n] = c.copy()
    x[n] = l
    if i % 5 == 2:
        continue
    x[i // 5] += r * (i - 2)

    while True:
        G = g_vec(x, A, b, c, r)
        if max(abs(G)) < eps:
            print((f(x, A, b, c, r), x.T, k))  # , g_func(x, c, r)

            break
        k += 1

        W = w_mat(x, A, c)
        # print(np.round(W, 10))
        W_inv = np.linalg.inv(W)
        # print(np.round(W_inv, 10))
        l_old = x[n, 0]
        x = x - W_inv.dot(G)
        # print(x.T, end=' ')

# print(mins, sep='\n')
