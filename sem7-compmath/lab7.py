# https://лови5.рф/upload/uf/477/uzvy38jk0t31titcv5nft2evy4qmv20x/CHislennoe-reshenie-uravneniya-Fredgolma-2-roda-metodom-kv.pdf
# quadrature method for Fredholm eq type 2
import numpy as np
import matplotlib.pyplot as plt


def K(x, s):
    return np.pi + np.sin(2 * (x + s))


def f(x):
    return x * np.cos(x) - 28 * np.pi * np.cos(2 * x)


def exact(x):
    return x * np.cos(x)


l = 7

a, b = -3 * np.pi, 3 * np.pi
n = 1000

xl, h = np.linspace(a, b, n + 1, retstep=True)

matrix = np.identity(n + 1)
F = np.zeros(n + 1) + f(xl)
w = l * np.ones(n + 1)
w[0] = w[-1] = l / 2

for i in range(n + 1):
    print(f"\r{i}", end="")
    Ki = lambda s: K(xl[i], s)
    matrix[i] -= w * Ki(xl) * h

# print(matrix, F)

u = np.linalg.solve(matrix, F)
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
