# simple iteration for Volterra eq type 2
import numpy as np
import matplotlib.pyplot as plt


def K(x, s):
    return np.sin(s - x)
    # return 1


def f(x):
    return np.cos(x) + 0.125 * x ** 2 * np.cos(x) - 0.125 * x * np.sin(x)
    # return 1


def exact(x):
    return np.cos(x) - 0.5 * x * np.sin(x)
    # return np.exp(x)


l = 1

a, b = 0, 3 * np.pi
# a, b = 0, 1


def solve(n=100, eps=1e-5, max_iter=10000):
    xl, h = np.linspace(a, b, n + 1, retstep=True)

    fx = np.zeros(n + 1) + f(xl)
    u = fx.copy()
    u_prev = np.zeros_like(u)

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

    return xl, u_solve


def plot(n=100, eps=1e-5):
    xl, u = solve(n, eps)
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

    plt.show()


def error_plot():
    epss = [1e-3, 1e-4, 1e-5, 1e-10, 1e-13]
    for eps in epss:
        x, u = solve(100, eps)
        ue = exact(x)

        print("", max(abs(u - ue)))


plot(1000, eps=1e-13)
# error_plot()
