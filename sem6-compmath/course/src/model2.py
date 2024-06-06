import numpy as np
import math
import matplotlib.pyplot as plt


def runge_kutta(function, y0: float, a: float, b: float, h: float):
    num = math.ceil((b - a) / h)
    x_a = np.linspace(a, b, num=num, endpoint=False)
    y_a = [y0] * (num)

    for i in range(num - 1):
        k0 = function(x_a[i], y_a[i])
        k1 = function(x_a[i] + h / 2, y_a[i] + h * k0 / 2)
        k2 = function(x_a[i] + h / 2, y_a[i] + h * k1 / 2)
        k3 = function(x_a[i] + h, y_a[i] + h * k2)
        y_a[i + 1] = y_a[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)

    return x_a, np.array(y_a)


ksi1, ksi2, ksi3 = 10, 0, 0
a12, a13, a23 = 6, 2, 0.5
k12, k13, k23 = 4, 1, 0.5


def right(x, y):
    return np.array([
        (-1 * x + 10) * x - 1 * y * x,
        (1 * x - 5) * y,
    ])


def plotvec(lin1, lin2):
    grid = np.meshgrid(lin1, lin2)
    vec_grid = right(grid[0], grid[1])

    plt.streamplot(*grid, *vec_grid, color='black')





a, b = 0, 10
n = 10000
h = (b - a) / n

x0 = np.array([10, 10, 1])
# x0 = static_point()
# tl, xl = runge_kutta(right, x0, a, b, h)
# xl = xl.T

print(x0)

# plot3(tl, xl)

# plotp(tl, xl, 0, 1)
# plotp(tl, xl, 0, 2)
# plotp(tl, xl, 1, 2)

# plotp3(tl, xl)


plotvec(np.linspace(0, 10, 100), np.linspace(0, 10, 100))
# plotvec3(np.linspace(0.01, 40, 21), np.linspace(0.01, 40, 21), 0)

# print(x0:=static_point(4))
# a2 = k12 * a12**2 * x0[0] * x0[1] + k13 * a13**2 * x0[0]*x0[2] + k23 * a23 ** 2 * x0[1] * x0[2]
# print(f"{a2=}")
# a3 = x0[0] * x0[1] * x0[2] * a12 * a13* a23 * (k13- k12 * k23)
# print(f"{a3=}")


plt.show()
