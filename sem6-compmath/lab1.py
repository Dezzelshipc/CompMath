import numpy as np
import math
from enum import Enum
import matplotlib.pyplot as plt


def euler(function, y0: float, a: float, b: float, h: float):
    num = math.ceil((b - a) / h + 1)
    x_a = np.linspace(a, b, num=num)
    y_a = [y0] * num

    for i in range(num - 1):
        y_a[i + 1] = y_a[i] + h * function(x_a[i], y_a[i])

    return x_a, np.array(y_a)


def euler_recalc(function, y0: float, a: float, b: float, h: float):
    num = math.ceil((b - a) / h + 1)
    x_a = np.linspace(a, b, num=num)
    y_a = [y0] * num

    for i in range(num - 1):
        y_r = y_a[i] + h * function(x_a[i], y_a[i])
        y_a[i + 1] = y_a[i] + h * function(x_a[i], y_r)

    return x_a, np.array(y_a)


def runge_kutta(function, y0: float, a: float, b: float, h: float):
    num = math.ceil((b - a) / h + 1)
    x_a = np.linspace(a, b, num=num)
    y_a = [y0] * num

    for i in range(num - 1):
        k0 = function(x_a[i], y_a[i])
        k1 = function(x_a[i] + h / 2, y_a[i] + h * k0 / 2)
        k2 = function(x_a[i] + h / 2, y_a[i] + h * k1 / 2)
        k3 = function(x_a[i] + h, y_a[i] + h * k2)
        y_a[i + 1] = y_a[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)

    return x_a, np.array(y_a)


def plot_funcs(f, y0: float, a: float, b: float, h: float):
    x, y_e = euler(f[0], y0, a, b, h)
    _, y_er = euler_recalc(f[0], y0, a, b, h)
    _, y_rk = runge_kutta(f[0], y0, a, b, h)

    y_f = f[1](x)

    fig, (ax1, ax2) = plt.subplots(1, 2)

    x_f = np.linspace(a, b, num=10000)

    ax1.plot(x_f, f[1](x_f), 'g')
    ax1.plot(x, y_e, 'b')
    ax1.plot(x, y_er, 'c')
    ax1.plot(x, y_rk, 'r')

    ax2.plot(x, np.abs(y_f - y_e), 'b')
    ax2.plot(x, np.abs(y_f - y_er), 'c')
    ax2.plot(x, np.abs(y_f - y_rk), 'r')

    ax1.set_title("Functions")
    ax2.set_title("Error")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2.set_xlabel(f"{h=}")

    ax1.legend(['Func', 'Euler', 'Euler Recalc', 'Runge-Kutta'])

    plt.draw()


def plot_funcs_h(f, y0: float, a: float, b: float, h: float):
    n = int(-np.log2(h))
    h_a = np.logspace(0, -n, base=2, num=n)

    y_e_e = [0] * n
    y_er_e = [0] * n
    y_rk_e = [0] * n

    for i in range(n):
        hi = h_a[i]

        x, y_e = euler(f[0], y0, a, b, hi)
        _, y_er = euler_recalc(f[0], y0, a, b, hi)
        _, y_rk = runge_kutta(f[0], y0, a, b, hi)
        y_f = f[1](x)

        y_e_e[i] = np.max(np.abs(y_f - y_e))
        y_er_e[i] = np.max(np.abs(y_f - y_er))
        y_rk_e[i] = np.max(np.abs(y_f - y_rk))

    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.plot(h_a, y_e_e, 'b')
    ax1.plot(h_a, y_er_e, 'c')
    ax1.plot(h_a, y_rk_e, 'r')

    ax1.set_title("Errors Log")
    ax2.set_title("Errors Normal")

    ax1.legend(['Euler', 'Euler Recalc', 'Runge-Kutta'])
    ax1.set_xlabel("h")
    ax1.set_ylabel("max abs diff")

    ax1.set_xscale('log')
    ax1.set_yscale('log')

    ax2.plot(h_a, y_e_e, 'b')
    ax2.plot(h_a, y_er_e, 'c')
    ax2.plot(h_a, y_rk_e, 'r')

    plt.draw()


class Test(Enum):
    NONE = -1
    ONE = 0
    X2 = 1
    SINX = 2
    SQRT = 3
    XY = 4
    COS2Y = 5
    LN = 6
    SINX2Y = 7


if __name__ == '__main__':

    y0 = 1
    x0, xn = 0, 5
    h = 0.0000237

    t = Test.SINX2Y

    match t:
        case Test.ONE:
            f = (
                lambda a, b: 1,
                lambda a: a + (y0 - x0)
            )

        case Test.X2:
            f = (
                lambda a, b: 2 * a,
                lambda a: a ** 2 + (y0 - x0 ** 2)
            )

        case Test.SINX:
            f = (
                lambda a, b: np.sin(a),
                lambda a: -np.cos(a) + (y0 + np.cos(x0))
            )

        case Test.SQRT:
            f = (
                lambda a, b: np.sqrt(a),
                lambda a: a ** 1.5 * 2 / 3 + (y0 - 2 / 3 * x0 ** 1.5)
            )

        case Test.XY:
            f = (
                lambda a, b: a + b,
                lambda a: (y0 + x0 + 1) / np.e ** x0 * np.e ** a - a - 1
            )

        case Test.COS2Y:
            f = (
                lambda a, b: np.cos(b) ** 2,
                lambda a: np.arctan(np.tan(y0) - x0 + a)
            )

        case Test.LN:
            f = (
                lambda a, b: 1 / a,
                lambda a: np.log(a) + (y0 + np.log(x0))
            )

        case Test.SINX2Y:
            sol = lambda x: np.e ** (0.5 * (x - np.sin(x) * np.cos(x)))

            f = (
                lambda a, b: (np.sin(a) ** 2) * b,
                lambda a: y0 / sol(x0) * sol(a)
            )

    plot_funcs(f, y0, x0, xn, h)

    plot_funcs_h(f, y0, x0, xn, h)

    plt.show()
