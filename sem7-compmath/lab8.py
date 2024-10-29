# singular(?) kernel method for Fredholm eq type 2
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.integrate import quad


def K(x, s):
    return x ** 2 * np.exp(x ** 2 * s ** 4)
    # return x*(np.exp(s*x) - 1)

# alpha
def Kx(i):
    return lambda x: x ** (2 + 2 * i)
    # return lambda x: x ** (2 + i)

# beta
def Ks(i):
    return lambda s: s ** (4 * i) / factorial(i)
    # return lambda s: s ** (1 + i) / factorial(i+1)


def f(x):
    return x ** 3 - np.exp(x ** 2) + 1
    # return np.exp(x) - x


def exact(x):
    return x ** 3
    # return 1

l = 4
# l = -1

a, b = 0, 1


def solve(n=3):
    F = np.zeros(n)
    A = np.zeros((n,n))

    for j in range(n):
        F[j] = quad(lambda s: f(s) * Ks(j)(s), a, b)[0]
        for i in range(n):
            A[j, i] = quad(lambda s: Kx(i)(s) * Ks(j)(s), a, b)[0]

    mat = np.eye(n) - l * A
    C = np.linalg.solve(mat, F)

    return lambda x: f(x) + l * sum(C[i] * Kx(i)(x) for i in range(n))


def plot(n = 3, xn = 1000):
    xl = np.linspace(a, b, xn+1)

    u = solve(n)
    e = np.zeros(xn+1) + exact(xl)
    u_v = u(xl)

    plt.figure("Решение")
    plt.plot(xl, u_v)
    plt.plot(xl, e, "--")
    plt.legend(["Численное", "Точное"])
    plt.title(f"{n = }")
    plt.grid()

    plt.figure("Модуль ошибки")
    plt.plot(xl, abs(u_v - e))
    plt.grid()

    plt.show()


def error_plot():
    xn = 1000
    xl = np.linspace(a, b, xn + 1)

    ue = np.zeros(xn + 1) + exact(xl)

    nl = [1, 2, 3, 5, 10, 20, 50, 100]
    for n in nl:
        u = solve(n)
        u_v = u(xl)

        print(n, max(abs(u_v - ue)))


# plot(10)
error_plot()
