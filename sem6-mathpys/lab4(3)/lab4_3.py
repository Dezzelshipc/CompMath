import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, cos, pi, sin, sinh


def f(x):
    return cos(x) - exp(x)


def A(n):
    if n == 0:
        return (1 - exp(2 * pi)) / (2 * pi)
    elif n == 1:
        return (1 + (1 - exp(2 * pi))) / (2 * pi)
    else:
        return (1 - exp(2 * pi)) / (2 * pi * (n ** 2 + 1))


def B(n):
    if n == 0:
        return 0
    elif n == 1:
        return (exp(2 * pi) - 1) / (2 * pi)
    else:
        return n * (exp(2 * pi) - 1) / (2 * pi * (n ** 2 + 1))


def A2(n):
    if n == 0:
        return (exp(-pi) - exp(pi)) / (2 * pi)
    elif n == 1:
        return 1 + sinh(pi) / pi
    else:
        return -(2 * sinh(pi) * cos(pi * n)) / (pi * (n ** 2 + 1))


def B2(n):
    if n == 0:
        return 0
    else:
        return (2 * n * sinh(pi) * cos(pi * n)) / (pi * (n ** 2 + 1))


def F(x, n):
    return A(0) + sum(A(i) * cos(i * x) + B(i) * sin(i * x) for i in range(1, n + 1))

def F2(x, n):
    return A2(0) + sum(A2(i) * cos(i * x) + B2(i) * sin(i * x) for i in range(1, n + 1))


x = np.linspace(0, 2*pi, 10000)
# x = np.linspace(-pi, pi, 10000)

plt.plot(x, F(x, 100))
plt.plot(x, f(x))
plt.legend(["Ряд Фурье", "Точное"])

plt.tight_layout()
# plt.savefig('./sem6-mathpys/lab4(3)/plot1.pdf')
plt.show()
