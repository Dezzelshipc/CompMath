import numpy as np
import matplotlib.pyplot as plt


def runge_kutta(function, y0: float, a: float, b: float, h: float):
    num = int(np.ceil((b - a) / h))
    x_a = np.linspace(a, b, num=num, endpoint=False)
    y_a = [y0] * num

    for i in range(num - 1):
        k0 = function(x_a[i], y_a[i])
        k1 = function(x_a[i] + h / 2, y_a[i] + h * k0 / 2)
        k2 = function(x_a[i] + h / 2, y_a[i] + h * k1 / 2)
        k3 = function(x_a[i] + h, y_a[i] + h * k2)
        y_a[i + 1] = y_a[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)

    return x_a, np.array(y_a)


def right(t, x):
    return np.array([
        x[2], # u, dx
        x[3], # v, dy
        2 * w * x[3], # du, ddx
        -2* w * x[2], # dv, ddy
    ])


t0, tn = 0, 5
n = 100000

w = 10
x0 = [5,3,0,6]

t, x = runge_kutta(right, x0, t0, tn, (tn-t0)/n)
