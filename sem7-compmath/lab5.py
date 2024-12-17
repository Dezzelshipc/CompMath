# https://лови5.рф/upload/uf/09b/vraz7xspskir759cukwoqan1sedpcnt8/CHislennoe-reshenie.pdf
# quadrature method for Volterra eq type 2
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
    return 0 * x


l = 1

# a, b = 0, 3 * np.pi
a, b = 0, 1

def solve(n=100, retstep=False):
    xl, h = np.linspace(a, b, n + 1, retstep=True)

    F = np.zeros(n + 1) + f(xl)
    A = l * np.ones(n + 1)
    A[0] = A[-1] = l / 2

    u = np.zeros(n + 1)

    for i in range(n + 1):
        print(f"\r{i}", end="")
        Ki = lambda s: K(xl[i], s)
        u[i] = (F[i] + sum((A * Ki(xl) * h * u))) / (1 - l*h/2 * Ki(xl[i]))

    if retstep:
        return xl, u, h
    return xl, u


def find_nev(u, xl, h):
    un = np.zeros_like(u)
    for j in range(1, len(un)):
        Kj = lambda s: K(xl[j], s)
        un[j] += Kj(xl[0]) * u[0] + Kj(xl[j]) * u[j]

        un[j] += 2 * sum(Kj(xl[1:j]) * u[1:j])

        un[j] = h / 2 * un[j]

    return u - l * un - f(xl)

def plot(n=100):
    xl, u, h = solve(n, retstep=True)
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
    nl = [10, 20, 50, 100, 500, 1000, 1e4]
    for n in nl:
        x, u, h = solve(int(n), retstep=True)
        ue = exact(x)

        print("", max(abs(u - ue)), max(abs(find_nev(u, x, h))))


if __name__ == "__main__":
    # plot(10)
    error_show()
