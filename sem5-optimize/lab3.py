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
            opt_val = np.zeros(n + m)
            opt_val[base_var] = lim.flatten()
            return max_val[0, 0], opt_val[:n]

        r_vec = lim / mat[:, k]
        r_vec = np.array([l if l > 0 else max(r_vec) + 1 for l in r_vec])
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
            opt_val = np.zeros(n + 2 * m)
            opt_val[base_var] = lim.flatten()
            return max_val[0, 0], opt_val[:n]

        r_vec = lim / mat[:, k]
        r_vec = np.array([l if l > 0 else max(r_vec) + 1 for l in r_vec])
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


if __name__ == "__main__":
    A = np.matrix([[2., 4.],
                   [1., 1.],
                   [2., 1.]])
    c = np.matrix([4., 5.])
    b = np.matrix([560., 170., 300.])

    # A = np.matrix([[4., 5., 1., 2., 2., 5.],
    #                [2., 9., 3., 5., 3., 6.],
    #                [7., 3., 6., 6., 2., 5.],
    #                [6., 1., 3., 3., 4., 6.],
    #                [6., 3., 8., 9., 6., 7.],
    #                [4., 6., 2., 3., 4., 7.],
    #                [9., 1., 4., 8., 5., 9.],
    #                [2., 3., 5., 2., 6., 6.]])
    # b = np.matrix([[819., 966., 822., 653., 649., 960., 734., 568.]])
    # c = np.matrix([[7., 2., 1., 6., 9., 2.]])

    # A = np.matrix([[1., 5., 9., 3., 2., 5.],
    #                [2., 9., 3., 5., 3., 6.],
    #                [7., 4., 6., 10., 2., 5.],
    #                [5., 9., 3., 3., 4., 6.],
    #                [6., 3., 8., 9., 6., 7.],
    #                [4., 6., 2., 3., 4., 7.],
    #                [9., 1., 2., 3., 5., 9.],
    #                [2., 3., 5., 2., 6., 10.]])
    # b = np.matrix([[630., 366., 722., 653., 439., 960., 734., 168.]])
    # c = np.matrix([[7., 2., 1., 6., 9., 2.]])

    # np.random.seed(12)
    # A = np.matrix(np.random.randint(1, 10, (8, 6))).astype(float)
    # c = np.matrix(np.random.randint(1, 10, (1, 6))).astype(float)
    # b = np.matrix(np.random.randint(500, 1000, (1, 8))).astype(float)

    # print(A, b, c, sep='\n')
    print(simplex(A, b, c))
    print(simplex_dual(A, b, c))
