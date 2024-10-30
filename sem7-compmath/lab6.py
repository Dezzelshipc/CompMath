# simple iteration for Volterra eq type 2
import numpy as np
import matplotlib.pyplot as plt


def K(x, s):
    # return np.sin(s - x)
    # return 1
    return (1 + s) * (np.exp(0.2 * s * x) - 1)


def f(x):
    # return np.cos(x) + 0.125 * x ** 2 * np.cos(x) - 0.125 * x * np.sin(x)
    # return 1
    return np.exp(-x)


def exact(x):
    # return np.cos(x) - 0.5 * x * np.sin(x)
    # return np.exp(x)
    return 0*x


l = 1

# a, b = 0, 3 * np.pi
a, b = 0, 1


def solve(n=100, eps=1e-5, max_iter=10000, retstep=False):
    xl, h = np.linspace(a, b, n + 1, retstep=True)

    fx = np.zeros(n + 1) + f(xl)
    u = fx.copy()

    u_solve = fx.copy()

    i = 0
    while i < max_iter and (err := np.max(abs(u)) / np.max(abs(u_solve - u) + [1e-5])) > eps:

        i += 1
        print(f"\r{i, err}", end="")
        u_prev = u.copy()
        u[:] = 0

        for j in range(n + 1):
            Kj = lambda s: K(xl[j], s)
            if j != 0:
                u[j] += Kj(xl[0]) * u_prev[0] + Kj(xl[j]) * u_prev[j]

            u[j] += 2 * sum(Kj(xl[1:j]) * u_prev[1:j])

            u[j] = h / 2 * u[j]

        u_solve += u

    if retstep:
        return xl, u_solve, h
    return xl, u_solve


def find_nev(u, xl, h):
    un = np.zeros_like(u)
    for j in range(len(un)):
        Kj = lambda s: K(xl[j], s)

        un[j] += Kj(xl[0]) * u[0] + Kj(xl[j]) * u[j]

        un[j] += 2 * sum(Kj(xl[1:j]) * u[1:j])

        un[j] = h / 2 * un[j]

    return u - l * un - f(xl)


def plot(n=100, eps=1e-5):
    xl, u, h = solve(n, eps, retstep=True)
    e = exact(xl)

    plt.figure("Решение")
    plt.plot(xl, u)
    plt.plot(xl, e, "--")
    plt.legend(["Численное", "Точное"])
    plt.title(f"{n = }")
    plt.grid()

    plt.figure("Модуль ошибки")
    plt.plot(xl, abs(u - e))
    plt.grid()

    plt.figure("Модуль невязки")
    plt.plot(xl, abs(find_nev(u, xl, h)))
    plt.grid()

    plt.show()


def error_show():
    epss = [1e-3, 1e-4, 1e-5, 1e-10, 1e-13]
    for eps in epss:
        x, u, h = solve(100, eps, retstep=True)
        ue = exact(x)

        print("", max(abs(u - ue)), max(abs(find_nev(u, x, h))))

def test_5_6():
    n = 100
    from lab5 import solve as solve5
    x, u5, h = solve5(n, retstep=True)
    _, u6 = solve(n)

    plt.figure("Решения")
    plt.plot(x, u5)
    plt.plot(x, u6, "--")
    plt.legend(["Квадратура", "Итерация"])
    plt.title(f"{n = }")

    plt.figure("Модуль разности")
    plt.plot(x, abs(u5 - u6))
    plt.grid()

    plt.show()

if __name__ == "__main__":
    # plot(100, eps=1e-5)
    error_show()
    test_5_6()