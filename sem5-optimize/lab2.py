import numpy as np


def f0(var_vec: np.ndarray, mat: np.matrix, b_vec: np.ndarray) -> int | float:
    vv = var_vec[:-1]
    out = vv.T.dot(mat).dot(vv) / 2 + vv.T.dot(b_vec)
    return out.sum()


def f(var_vec: np.ndarray, mat: np.matrix, b_vec: np.ndarray, center: np.ndarray, radius: float) -> int | float:
    n = len(mat)
    vv = var_vec[:n]
    out = vv.T.dot(mat).dot(vv) / 2 + vv.T.dot(b_vec) + var_vec[n] * (
            np.linalg.norm(vv[:n] - center) ** 2 - radius ** 2)
    return out.sum()


def f_grad(var_vec: np.ndarray, mat: np.matrix, b_vec: np.ndarray, center: np.ndarray) -> np.ndarray:
    n = len(mat)
    vv = var_vec[:n]
    return mat.dot(vv) + b_vec + 2 * var_vec[n, 0] * (vv - center)


def g_func(var_vec: np.ndarray, center: np.ndarray, radius: float):
    return np.linalg.norm(var_vec[:-1] - center) ** 2 - radius ** 2


def w_mat(vv: np.ndarray, mat: np.matrix, center: np.ndarray) -> np.matrix:
    n = len(mat)
    W = np.zeros((n + 1, n + 1))
    W[:n, :n] = mat + 2 * vv[n, 0] * np.eye(n)

    W[n, :n] = (2 * (vv[:-1] - center)).flatten()
    W[:n, n] = (2 * (vv[:-1] - center)).flatten()
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
# b = np.matrix([[1.], [1.], [1.], [1.]])
# c = np.matrix([[1.], [1.], [1.], [1.]])
# r = 2
# l = 1

# A = np.matrix([[-8, 2, 2, 3],
#                [2, -1, 1, 5],
#                [2, 1, 4, 3],
#                [3, 5, 3, 6]])
#
# b = np.matrix([2., 6., 5., 3.]).T
# c = np.matrix([5., 5., 5., 2.]).T
# r = 2
# l = 10

A = np.array([[5, 2, 1, 3],
         [2, 6, 4, 7],
         [1, 4, 10, 8],
         [3, 7, 8, 1]])
b = np.array([6, 1, 7, 2]).reshape([4, 1]).astype(float)
c = np.array([1, 1, 3, 1]).reshape([4, 1]).astype(float)
r = 10
l = r

# A = np.matrix([[5., 1., 0., 10.],
#                [1., 4., 2., 4.],
#                [0., 2., 1., 20.],
#                [10., 4., 20., 0.]])
# b = np.matrix([0., 1., 1., 1.]).T
# c = np.matrix([0., 0., 0., 0.]).T
# l = 5.
# r = 10

# A = np.matrix([[8., 3., 6., 6.],
#                [3., 4., 5., 5.],
#                [6., 5., 4., 7.],
#                [6., 5., 7., 6.]])
# b = np.matrix([2., 6., 5., 3.]).T
# l = 1.
# c = np.matrix([5., 5., 5., 2.]).T
# r = 1

for i in range(len(A)):
    print(np.linalg.det(A[:i + 1, :i + 1]), end=' ')
print()

n = len(A)

eps = 1e-10

xs = []
for index in range(n):
    for koef in [-1.5, -1, -0.5, 0.5, 1, 1.5]:
        new_x = c.copy()
        new_x[index] += koef * r
        xs.append(new_x)

# xs = [np.matrix([1,2,3,4]).T]
mins = []
x = np.matrix([[0.]] * (n + 1))
for x_i in xs:
    x[:n] = x_i
    x[n] = l

    k = 0
    while True:
        G = g_vec(x, A, b, c, r)
        if max(abs(G)) < eps or k > 1000:
            mins.append((f0(x, A, b), x.copy().T, k))  # , g_func(x, c, r)

            break
        k += 1

        W = w_mat(x, A, c)
        # print(np.round(W, 10))
        W_inv = np.linalg.inv(W)
        # print(np.round(W_inv, 10))
        x = x - W_inv.dot(G)
        # print(x.T, end=' ')

print(*mins, sep='\n')
print()

# l == 0
A_inv = np.linalg.inv(A)
x = - A_inv.dot(b)
x = np.append(np.array(x), [0]).T
g_num = g_func(x, c, r)
if g_num < 0:
    print(f'При l = 0 решение подходит: {x = }, f0 = {f0(x, A, b)}, g = {g_num}')
else:
    print(f'При l = 0 решение не подходит: {x = }, f0 = {f0(x, A, b)}, g = {g_num}')

# mins = filter(mins, lambda y: y[1][-1] > 0)
print(f"MINMIN: {min(mins, key=lambda y: y[0])}")
