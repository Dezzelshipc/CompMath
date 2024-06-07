# FEM
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import sympy as sy


# var 25
def exact(x):
    # return np.log(x**2 + x + 1)
    return np.sqrt(x + 1) * np.log(x + 1)


def p(x):
    # return 1 / (x**2 + x + 1)
    return 2 * np.sqrt(x + 1) / (x + 1)
    # return 0


def q(x):
    # return 0
    return -1 / (np.sqrt(x + 1) * (x + 1))
    # return (1+x**2)


def f(x):
    # return (2 - 2*x**2)/(x**2 + x + 1)**2
    return (2 - np.log(x + 1) / (4 * np.sqrt(x + 1))) / (x + 1)
    # return -1


def F(x):
    return f(x) - p(x) * vp(x) - q(x) * v(x)


def v(x):
    return ua + (ub - ua) * (x - a) / (b - a)


def vp(x):
    return (ub - ua) / (b - a)


def base_phi(x):
    return np.where(abs(x) <= 1, np.where(x <= 0, 1 + x, 1 - x), 0)


def phi(x, k):
    return base_phi((x - xl[k]) / h)


def u(x, c):
    return sum(c[i] * phi(x, i + 1) for i in range(len(c))) + v(x)


a, b = 0, 1
ua, ub = exact(a), exact(b)
n = 30

xl, h = np.linspace(a, b, n + 2, retstep=True)
# print(xl)

matrix = np.zeros((n, n))
values = np.zeros(n)

for i in range(1, n + 1):

    if i > 1:
        intm = sc.integrate.quad(
            lambda x: p(x) * (x - xl[i - 1]) + q(x) * (x - xl[i - 1]) * (x - xl[i]),
            xl[i - 1], xl[i])
        matrix[i - 2, i - 1] = 1 / h - 1 / h ** 2 * intm[0]

    if i < n:
        intm = sc.integrate.quad(
            lambda x: p(x) * (x - xl[i + 1]) + q(x) * (x - xl[i]) * (x - xl[i + 1]),
            xl[i], xl[i + 1])
        matrix[i, i - 1] = 1 / h - 1 / h ** 2 * intm[0]

    int1 = sc.integrate.quad(
        lambda x: p(x) * (x - xl[i - 1]) + q(x) * (x - xl[i - 1]) ** 2,
        xl[i - 1], xl[i])

    int2 = sc.integrate.quad(
        lambda x: p(x) * (x - xl[i + 1]) + q(x) * (x - xl[i + 1]) ** 2,
        xl[i], xl[i + 1])

    matrix[i - 1, i - 1] = -2 / h + 1 / h ** 2 * (int1[0] + int2[0])

    intv1 = sc.integrate.quad(lambda x: F(x) * (x - xl[i - 1]), xl[i - 1], xl[i])
    intv2 = sc.integrate.quad(lambda x: F(x) * (x - xl[i + 1]), xl[i], xl[i + 1])
    values[i - 1] = (intv1[0] - intv2[0]) / h

# print(matrix)
# print(values)
c = np.linalg.solve(matrix.T, values)

print(c)
num = 10000
x = np.linspace(a, b, num + 1)

uu = u(x, c)
print(uu)
plt.plot(x, uu, 'r')

plt.plot(x, exact(x), 'b--')
plt.legend(["Решение", "Точное"])
plt.grid()
plt.plot(xl, [0] * len(xl), 'o')

plt.figure(123)
udiff = abs(exact(x) - uu)

plt.plot(x, udiff)

plt.show()
