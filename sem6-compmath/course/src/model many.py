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


# def static_point(num: int, num2=None):
#     match (num, num2):
#         case (0, None):
#             return [0, 0, 0]
#         case (1, None) | (1, 2) | (2, 1):
#             return [0, ksi3 / (k23 * a23), ksi2 / a23]
#         case (2, None) | (0, 2) | (2, 0):
#             return [ksi3 / (k13 * a13), 0, ksi1 / a13]
#         case (3, None) | (0, 1) | (1, 0):
#             return [-ksi2 / (k12 * a12), ksi1 / a12, 0]
#         case (4, None):
#             ld = k13 - k12 * k23
#             if ld != 0:
#                 return np.array([
#                     (ksi3 * a12 - ksi1 * a23 * k23 + ksi2 * k23 * a13) / (a12 * a13 * ld),
#                     (ksi1 * a13 * k13 - ksi2 * a13 * k13 - ksi3 * a12 * k12) / (a12 * a23 * ld),
#                     (ksi3 * a12 * k12 + ksi2 * a13 * k13 - ksi1 * k23 * k12 * k23) / (a13 * a23 * ld),
#                 ])
#             else:
#                 x3 = 1
#                 return np.array([
#                     (a23 * x3 - ksi2) / (a12 * k12),
#                     (-a13 * x3 + ksi1) / a12,
#                     x3,
#                 ])


# ksi1, ksi2, ksi3 = 10, 8, 6
# a12, a13, a23 = 6, 2, 0.5
# k12, k13, k23 = 4, 1, 0.5


# def right(t, x):
#     return np.array([
#         (ksi1 - a12 * x[1] - a13 * x[2]) * x[0],
#         (ksi2 + k12 * a12 * x[0] - a23 * x[2]) * x[1],
#         (-ksi3 + k13 * a13 * x[0] + k23 * a23 * x[1]) * x[2]
#     ])


def right(t, x):
    return np.array([
        ((-1 * x[0] + 10) * x[0] - 2 * x[1] * x[0] - 3 * x[2] * x[0]),
        ((1 * x[0] - 5) * x[1] - 1 * x[2] * x[1]),
        ((1 * x[0] - 3) * x[2] + (1 * x[1] - 4) * x[2])
    ])


def plotp(tl, xls, n1, n2):
    plt.figure(f"{n1}{n2}")
    plt.xlabel(f"x{n1 + 1}")
    plt.ylabel(f"x{n2 + 1}")
    # st_point = static_point(n1, n2)
    # plt.plot(st_point[n1], st_point[n2], 'ok')

    leg = [] #[np.array(st_point)]
    for xl in xls:
        plt.plot(xl[n1], xl[n2], 'o-', markevery=[0])
        leg.append(xl[:, 0])

    plt.legend([f"{pt}" for pt in leg])


def plotp3(tl, xls):
    ax = plt.figure("123").add_subplot(projection='3d')

    ax.set_xlabel('$x_1$')
    ax.set_ylabel('$x_2$')
    ax.set_zlabel('$x_3$')

    # ax.plot(*static_point(0), 'ok')
    # ax.plot(*static_point(1), 'or')
    # ax.plot(*static_point(2), 'og')
    # ax.plot(*static_point(3), 'om')
    # ax.plot(*static_point(4), 'om')

    ax.legend(["x(0)", "x(1)", "x(2)", "x(3)", "x(4)"])

    for xl in xls:
        ax.plot(xl[0], xl[1], xl[2], 'o-', markevery=[0])


def sol_many(function, y0s: list, a: float, b: float, h: float):
    sols = []
    for y0 in y0s:
        tl, xl = runge_kutta(function, y0, a, b, h)
        sols.append(xl.T)

    return tl, np.array(sols)


a, b = 0, 10
n = 10000
h = (b - a) / n

# x0s = [
#     [10, 10, 10],
#     [0, 60, 10],
#     [10, 0, 10],
#     [10, 10, 20],
#     [5, 9, 11]
# ]


x0s = [
    [10, 10, 10],
    [0, 6, 1],
    [1, 0, 1],
    [1, 1, 2],
    [5, 5, 5],
    [11, 0, 0],
    [11, 0, 1],
    [10, 1, 0],
    [3, 10, 0]
]

# x0s = [
#     [0, 100, 40],
#     [0, 50, 30],
#     [0, 50, 50],
#     [0, 30, 20],
#     [0, 20, 50]
# ]

# x0s = [
#     [10, 0, 20],
#     [10, 0, 10],
#     [5, 0, 5],
#     [4, 0, 2],
#     [15, 0, 6]
# ]

# x0s = [
#     [10, 20, 0],
#     [10, 10, 0],
#     [5,  5, 0],
#     [4,  2, 0],
#     [15, 6, 0],
# ]


tl, xls = sol_many(right, x0s, a, b, h)

plotp(tl, xls, 0, 1)
plotp(tl, xls, 0, 2)
plotp(tl, xls, 1, 2)

plotp3(tl, xls)

plt.show()
