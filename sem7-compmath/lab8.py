# singular(?) kernel method for Fredholm eq type 2
import numpy as np
import matplotlib.pyplot as plt
from math import factorial
from scipy.integrate import quad


def K(x, s):
    # return x ** 2 * np.exp(x ** 2 * s ** 4)
    # return x*(np.exp(s*x) - 1)
    # return (1 + s) * (np.exp(0.2 * s * x) - 1)
    return x * (np.sin(x*s) - 1)

# alpha
def Kx(i):
    # return lambda x: x ** (2 + 2 * i)
    # return lambda x: x ** (2 + i)
    # return lambda x: x ** (1 + i)
    return lambda x: x**(2*i) if i > 0 else x

# beta
def Ks(i):
    # return lambda s: s ** (4 * i) / factorial(i)
    # return lambda s: s ** (1 + i) / factorial(i+1)
    # return lambda s: s ** (1 + i) * (s+1) * 0.2**(i+1) / factorial(i+1)
    return lambda s: (-1)**(i+1) * s**(2*i - 1) / factorial(2*i-1) if i > 0 else -1


def f(x):
    # return x ** 3 - np.exp(x ** 2) + 1
    # return np.exp(x) - x
    # return np.exp(-x)
    return x + np.cos(x)


def exact(x):
    # return x ** 3
    # return 1 + 0 * x
    # return 0 * x
    return 1 + 0 * x


# l = 4
# l = -1
# l = 1
l = 1

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

def find_nev(u, xl, h):
    def nev_int(x):
        Kk = lambda s: K(x, s)
        un = Kk(xl[0]) * u[0] + Kk(xl[-1]) * u[-1] + 2 * sum(Kk(xl[1:-1]) * u[1:-1])
        un = h / 2 * un
        return un

    nev_i = np.array([nev_int(x) for x in xl])

    return u - l * nev_i - f(xl)

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

    plt.figure("Сравнение")
    plt.plot(xl, abs(u_v - e))
    plt.grid()

    nevf = lambda x: u(x) - l * quad(lambda s: K(x, s) * u(s), a, b)[0] - f(x)
    nev = np.zeros_like(xl)
    for i, x in enumerate(xl):
        nev[i] = nevf(x)
    plt.plot(xl, nev, "--")

    plt.legend(["Ошибка", "Невязка"])

    plt.show()


def error_show():
    xn = 1000
    xl, h = np.linspace(a, b, xn + 1, retstep=True)

    ue = np.zeros(xn + 1) + exact(xl)

    nl = [1, 2, 3, 5, 10, 20, 50, 100]
    for n in nl:
        u = solve(n)
        u_v = u(xl)

        nevf = lambda x: u(x) - l * quad(lambda s: K(x, s) * u(s), a, b)[0] - f(x)
        nev = np.zeros_like(xl)
        for i, x in enumerate(xl):
            nev[i] = nevf(x)

        print(n, max(abs(u_v - ue)), max(abs(find_nev(u_v, xl, h))), max(abs(nev)))


def test_7_8():
    nk = 100
    ns = 10
    from lab7 import solve as solve7
    x, u7, h = solve7(nk, retstep=True)
    u8f = solve(ns)
    u8 = u8f(x)


    plt.figure("Решения")
    plt.plot(x, u7)
    plt.plot(x, u8, "--")
    plt.legend(["Квадратура", "Вырожденные ядра"])
    plt.title(f"{nk = }, {ns = }")

    plt.figure("Модуль разности")
    plt.plot(x, abs(u7 - u8))
    plt.grid()

    plt.show()


if __name__ == "__main__":
    plot(3)
    error_show()
    test_7_8()
