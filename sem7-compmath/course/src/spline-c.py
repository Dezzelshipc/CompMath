# 3-spline collocation
import numpy as np
import matplotlib.pyplot as plt

def exact(x):
    # return x + np.exp(x)
    # return np.sqrt(x+1) * np.log(x+1)
    # return np.sin(x)/np.sin(1) - x
    return -2 * np.sqrt(x+1)

def q(x):
    # return -1
    # return -1/(np.sqrt(x+1) * (x+1))
    # return 1
    return -2 / (x+1)**2

def f(x):
    # return -x
    # return (-1/(x+1) - 1/(4 * (x+1)**1.5)) * np.log(x+1)
    return 4.5 / (x+1)**(3/2)

a, b = 0, 1
# a0, b0, c0 = 1, 0, exact(a)
# a1, b1, c1 = 1, 0, exact(b)

a0, b0, c0 = 1, -2, 0
a1, b1, c1 = 0, 1, -1/np.sqrt(2)

n = 10
tl, h = np.linspace(a, b, n + 1, retstep=True)
mu = 0.5  # h/(h+h)

A = np.zeros(n + 1)
C = np.zeros(n + 1)
D = np.zeros(n + 1)
F = np.zeros(n + 1)

for k in range(1, n):
    A[k] = (1 - mu) * (1 + h ** 2 * q(tl[k - 1]) / 6)
    D[k] = mu * (1 + h ** 2 * q(tl[k + 1]) / 6)
    C[k] = -(1 - h ** 2 * q(tl[k]) / 3)
    F[k] = h ** 2 / 6 * (mu * f(tl[k - 1]) + 2 * f(tl[k]) + (1 - mu) * f(tl[k + 1]))

C[0] = a0 * h - b0 * (1 - 1 / 3 * q(tl[0]) * h ** 2)
D[0] = b0 * (1 + 1 / 6 * q(tl[1]) * h ** 2)
F[0] = c0 * h + 1 / 6 * b0 * h ** 2 * (2 * f(tl[0]) + f(tl[1]))

A[-1] = b1 * (-1 - 1 / 6 * h ** 2 * q(tl[-2]))
C[-1] = a1 * h + b1 * (1 - 1 / 3 * h ** 2 * q(tl[-1]))
F[-1] = c1 * h - 1 / 6 * b1 * h ** 2 * (f(tl[-2]) + 2 * f(tl[-1]))

matrix = np.diag(A[1:], -1) + np.diag(C) + np.diag(D[:-1], 1)

v = np.linalg.solve(matrix, F)

M = F - q(tl) * v

n_in = 100

xl = np.zeros(0)
uu = np.zeros(0)

for i in range(n):
    xl_cur = np.linspace(tl[i], tl[i + 1], n_in)

    def S(x):
        t = (x - tl[i]) / h
        return v[i] * (1 - t) + v[i + 1] * t - h ** 2 / 6 * t * (1 - t) * ((2 - t) * M[i] + (1 + t) * M[i + 1])

    u_cur = S(xl_cur)

    xl = np.append(xl, xl_cur)
    uu = np.append(uu, u_cur)

ex = exact(xl)

plt.figure("Решения")
plt.plot(xl, uu)
plt.plot(xl, ex, "--")
plt.legend(["Численное", "Точное"])
plt.grid()

plt.figure("Ошибка")
plt.plot(xl, abs(uu - exact(xl)))
plt.title(f"{n = }")
plt.grid()

print(max( abs(uu - exact(xl))))

plt.show()
