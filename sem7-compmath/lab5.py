# https://лови5.рф/upload/uf/09b/vraz7xspskir759cukwoqan1sedpcnt8/CHislennoe-reshenie.pdf
# quadrature method for Volterra eq type 2
import numpy as np
import matplotlib.pyplot as plt


def K(x, s):
    return np.sin(s - x)


def f(x):
    return np.cos(x) + 0.125 * x ** 2 * np.cos(x) - 0.125 * x * np.sin(x)


def exact(x):
    return np.cos(x) - 0.5 * x * np.sin(x)


l = 1

a, b = 0, 3 * np.pi

def solve(n=100):
    xl, h = np.linspace(a, b, n + 1, retstep=True)

    F = np.zeros(n + 1) + f(xl)
    A = l * np.ones(n + 1)
    A[0] = A[-1] = l / 2

    u = np.zeros(n + 1)

    for i in range(n + 1):
        print(f"\r{i}", end="")
        Ki = lambda s: K(xl[i], s)
        u[i] = (F[i] + sum((A * Ki(xl) * h * u))) / (1 - A[i] * Ki(xl[i]))

    return xl, u

def plot(n = 100):
    xl, u = solve(n)
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
    nl = [10, 20, 50, 100, 500, 1000, 1e4]
    for n in nl:
        x, u = solve(int(n))
        ue = exact(x)

        print("", max(abs(u - ue)))

# plot(10000)
error_plot()
