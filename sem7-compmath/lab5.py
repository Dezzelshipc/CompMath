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
n = 10

xl, h = np.linspace(a, b, n + 1, retstep=True)

F = np.zeros(n + 1) + f(xl)
w = l * np.ones(n + 1)
w[0] = w[-1] = l / 2

u = np.zeros(n + 1)

for i in range(n + 1):
    print(f"\r{i}", end="")
    Ki = lambda s: K(xl[i], s)
    u[i] = (F[i] + sum((w * Ki(xl) * h * u))) / (1 - w[i] * Ki(xl[i]))

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
