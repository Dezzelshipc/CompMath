import numpy as np


# c * x -> max; mat * x <= lim
def simplex(mat_orig: np.matrix, lim_orig: np.matrix, coef_orig: np.matrix):
    n = len(coef_orig.T)
    m = len(mat_orig)
    full_range = np.arange(m)
    mat = np.append(mat_orig, np.eye(m), axis=1)
    base_var = np.arange(n, m + n)
    lim = lim_orig.copy().T
    coef = np.append(coef_orig, np.zeros((1, m)), axis=1).T

    while True:
        delta = coef - mat.T.dot(coef[base_var, :])
        k = delta.argmax()
        if delta[k, :] <= 0:
            max_val = coef[base_var, :].T.dot(lim)
            return max_val[0, 0]

        r_vec = lim / mat[:, k]
        r = r_vec.argmin()

        new_mat = mat.copy()
        new_lim = lim.copy()
        new_mat[r, :] /= mat[r, k]
        new_lim[r, :] /= mat[r, k]
        for i in full_range[full_range != r]:
            new_mat[i] -= mat[i, k] * new_mat[r]
            new_lim[i] -= mat[i, k] * new_lim[r]

        base_var[r] = k
        mat = new_mat
        lim = new_lim


def simplex_dual(mat_orig: np.matrix, lim_orig: np.matrix, coef_orig: np.matrix):
    mat_orig = mat_orig.T.copy()
    lim_orig, coef_orig = coef_orig.copy(), lim_orig.copy()
    big_number = np.max(coef_orig) * 10
    n = len(coef_orig.T)
    m = len(mat_orig)
    full_range = np.arange(m)
    mat = np.append(mat_orig, np.append(-np.eye(m), np.eye(m), axis=1), axis=1)
    base_var = np.arange(n + m, n + 2 * m)
    lim = lim_orig.copy().T
    coef = np.append(coef_orig, np.append(np.zeros((1, m)), big_number * np.ones((1, m)), axis=1), axis=1).T

    while True:
        delta = coef - mat.T.dot(coef[base_var, :])
        k = delta.argmin()
        # print(delta.T)
        if delta[k, :] >= 0:
            max_val = coef[base_var, :].T.dot(lim)
            return max_val[0, 0]

        r_vec = lim / mat[:, k]
        r = r_vec.argmin()

        new_mat = mat.copy()
        new_lim = lim.copy()
        new_mat[r, :] /= mat[r, k]
        new_lim[r, :] /= mat[r, k]
        for i in full_range[full_range != r]:
            new_mat[i] -= mat[i, k] * new_mat[r]
            new_lim[i] -= mat[i, k] * new_lim[r]

        base_var[r] = k
        mat = new_mat
        lim = new_lim


# A = np.matrix([[2., 4.],
#                [1., 1.],
#                [2., 1.]])
# c = np.matrix([4., 5.])
# b = np.matrix([560., 170., 300.])

# A = np.matrix([[1., -1., -2., -1],
#                [2., 0., 1., -4.],
#                [-2., 1., 0., 1.]])
# c = np.matrix([1., -2., -3., -1.])
# b = np.matrix([4., 2., 1.])

np.random.seed(1000)
A = np.matrix(np.random.randint(0, 10, (8, 6))).astype(float)
c = np.matrix(np.random.randint(0, 10, (1, 6))).astype(float)
b = np.matrix(np.random.randint(500, 1000, (1, 8))).astype(float)


print(A, b, c, sep='\n')
print(simplex(A, b, c))
# print(simplex_dual(A, b, c))
