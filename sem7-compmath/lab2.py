# b-spline collocation
import numpy as np
import matplotlib.pyplot as plt


# var 25
def exact(x):
    return np.log(x ** 2 + x + 1)
    # return np.sqrt(x+1) * np.log(x+1)
    # return 0.988 * (1-x**2) - 0.0543 * (1-x**4)


def p(x):
    return 1 / (x ** 2 + x + 1)
    # return 2 * np.sqrt(x+1) / (x+1)
    # return 0


def q(x):
    return 0
    # return -1 / (np.sqrt(x+1) * (x+1))
    # return (1 + x ** 2)


def f(x):
    return (2 - 2 * x ** 2) / (x ** 2 + x + 1) ** 2
    # return (2 - np.log(x+1) / (4 * np.sqrt(x+1))) / (x+1)
    # return -1


def bs(t, i, d):
    if d == 0:
        return np.where(tl[i] <= t, np.where(t < tl[i + 1], 1, 0), 0)
    val = ((t - tl[i]) / (d * h) * bs(t, i, d - 1)
           + (tl[i + d + 1] - t) / (d * h) * bs(t, i + 1, d - 1))
    return val


def u(x, c):
    return sum(c[i] * bs(x, i - 2, 3) for i in range(-1, n + 2))


a, b = 0, 1
# a0, b0, c0 = 1, 0, exact(a)
# a1, b1, c1 = 1, 0, exact(b)

a0, b0, c0 = 1, 0, 0
a1, b1, c1 = 0, 1, 1

n = 1000
tl, h = np.linspace(a, b, n + 1, retstep=True)
tl = np.append(tl, [b + h, b + 2 * h, b + 3 * h, a - 3 * h, a - 2 * h, a - h])
print(tl, h)

A = np.zeros(n + 1)
D = np.zeros(n + 1)
C = np.zeros(n + 1)
F = np.zeros(n + 1)

for k in range(n + 1):
    A[k] = (1 - p(tl[k]) * h / 2 + q(tl[k]) * h ** 2 / 6) / (3 * h)
    D[k] = (1 + p(tl[k]) * h / 2 + q(tl[k]) * h ** 2 / 6) / (3 * h)
    C[k] = -A[k] - D[k] + q(tl[k]) * h / 3
    F[k] = f(tl[k]) * h / 3

Am1 = a0 * h - 3 * b0
Cm1 = 4 * a0 * h
Dm1 = a0 * h + 3 * b0
Fm1 = 6 * c0 * h

An1 = a1 * h - 3 * b1
Cn1 = 4 * a1 * h
Dn1 = a1 * h + 3 * b1
Fn1 = 6 * c1 * h

C[0] -= Cm1 * A[0] / Am1
D[0] -= Dm1 * A[0] / Am1
F[0] -= Fm1 * A[0] / Am1
A[-1] -= An1 * D[-1] / Dn1
C[-1] -= Cn1 * D[-1] / Dn1
F[-1] -= Fn1 * D[-1] / Dn1
# print(A,D,C,F)


M = np.diag(A[1:], -1) + np.diag(C) + np.diag(D[:-1], 1)
# print(M)
# print(F)

const = np.linalg.solve(M, F)

bm1 = (Fm1 - const[0] * Cm1 - const[1] * Dm1) / Am1
bn1 = (Fn1 - const[-1] * Cn1 - const[-2] * An1) / Dn1
const = np.append(const, [bn1, bm1])

print(const)

N = 10000
xl = np.linspace(a - h * 3, b + h * 3, N)
uu = u(xl, const)
ex = exact(xl)

plt.figure("Решение")
plt.plot(xl, uu)
plt.plot(xl, ex, "--")
plt.legend(["Численное", "Точное"])
plt.grid()

if 1:
    for i in range(-1, n + 2):
        plt.plot(xl, const[i] * bs(xl, i - 2, 3))

plt.figure("Модуль ошибки")
xl1 = np.linspace(a, b, N)
plt.plot(xl, abs(u(xl1, const) - exact(xl1)))
plt.title(f"{n = }")
plt.grid()

plt.show()
