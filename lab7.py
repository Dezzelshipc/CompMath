import matplotlib.pyplot as plt

import numpy as np


## Tri Diagonal Matrix Algorithm(a.k.a Thomas algorithm) solver
def TDMAsolver(a, b, c, d):
    '''
    TDMA solver, a b c d can be NumPy array type or Python list type.
    refer to http://en.wikipedia.org/wiki/Tridiagonal_matrix_algorithm
    and to http://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
    '''
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc


def TDMA(a, b, c, f):
    # a, b, c, f = tuple(map(lambda k_list: list(map(float, k_list)), (a, b, c, f)))

    alpha = [-b[0] / c[0]]
    beta = [f[0] / c[0]]
    n = len(f)
    x = [0] * n

    for i in range(1, n):
        alpha.append(-b[i] / (a[i] * alpha[i - 1] + c[i]))
        beta.append((f[i] - a[i] * beta[i - 1]) / (a[i] * alpha[i - 1] + c[i]))

    x[n - 1] = beta[n - 1]

    for i in range(n - 1, -1, -1):
        x[i - 1] = alpha[i - 1] * x[i] + beta[i - 1]

    print(alpha, beta, x)

    return x


def func(t):
    return 6 * t
    # return 0


def u(t):
    return t ** 3
    # return t


a, b = 0, 1
ua, ub = 0, 1
N = 3
h = 1 / N

under = [1 / h ** 2 for _ in range(N)]
over = [1 / h ** 2 for _ in range(N)]
middle = [-2 / h ** 2 for _ in range(N + 1)]
# fs = [ua, *(func(a + h * i) for i in range(N-1)), ub]
fs = [func(a + h * i) for i in range(N + 1)]
# fs = [1 for i in range(N + 1)]

print(under, over, middle, fs, sep='\n')

ans = TDMAsolver(under,
                 middle,
                 over,
                 fs)

# print(*(h * i for i in range(n)))
# print(ans)

x = [a + h * i for i in range(N + 1)]
print(x)
plt.plot(x, ans, 'ro', linestyle='-', markersize=0)
plt.plot(x, [u(i) for i in x], 'bo', linestyle='-', markersize=0)
plt.show()
