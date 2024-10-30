# https://лови5.рф/upload/uf/477/uzvy38jk0t31titcv5nft2evy4qmv20x/CHislennoe-reshenie-uravneniya-Fredgolma-2-roda-metodom-kv.pdf
# quadrature method for Fredholm eq type 2
import numpy as np
import matplotlib.pyplot as plt


def K(x, s):
    # return np.pi + np.sin(2 * (x + s))
    return (1 + s) * (np.exp(0.2 * s * x) - 1)


def f(x):
    # return x * np.cos(x) - 28 * np.pi * np.cos(2 * x)
    return np.exp(-x)


def exact(x):
    # return x * np.cos(x)
    return 0*x


# l = 7
l = 1

# a, b = -3 * np.pi, 3 * np.pi
a, b = 0, 1

def solve(n = 100, retstep=False):
    xl, h = np.linspace(a, b, n + 1, retstep=True)

    matrix = np.identity(n + 1)
    F = np.zeros(n + 1) + f(xl)
    A = l * np.ones(n + 1)
    A[0] = A[-1] = l / 2

    for i in range(n + 1):
        print(f"\r{i}", end="")
        Ki = lambda s: K(xl[i], s)
        matrix[i] -= A * Ki(xl) * h

    u = np.linalg.solve(matrix, F)

    if retstep:
        return xl, u, h
    return xl, u

def find_nev(u, xl, h):
    def nev_int(x):
        Kk = lambda s: K(x, s)
        un = Kk(xl[0]) * u[0] + Kk(xl[-1]) * u[-1] + 2 * sum(Kk(xl[1:-1]) * u[1:-1])
        un = h / 2 * un
        return un

    nev_i = np.array([nev_int(x) for x in xl])

    return u - l * nev_i - f(xl)

def plot(n = 100):
    xl, u = solve(n)
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

def error_show():
    nl = [10, 20, 50, 100, 500, 1000]
    # nl = [2, 5, 10]
    for n in nl:
        x, u, h = solve(int(n), retstep=True)
        ue = exact(x)

        print("", max(abs(u - ue)), max(abs(find_nev(u, x, h))))

if __name__ == "__main__":
    # plot(2)
    error_show()