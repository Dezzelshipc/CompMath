# switching directions method for parabolic pde (heat eq)
import time

import numpy as np
import matplotlib.pyplot as plt
import scipy


def exact(x, y, t):
    return t * np.exp(x + y)
    # return t * np.sin(np.pi * x) * np.sin(np.pi * y)
    # return t + (y ** 2 + x ** 2)
    # return t + (y ** 2 + x ** 2) / 4


def f(x, y, t):
    return (1 - 2 * t) * np.exp(x + y)
    # return (1 + 2 * t * np.pi ** 2) * np.sin(np.pi * x) * np.sin(np.pi * y)
    # return -3
    # return 0 * x * t * y


def l(size):
    lmat = -2 * np.identity(size + 1)
    ones1 = np.ones(size + 1)
    lmat += np.diag(ones1[1:], -1) + np.diag(ones1[:-1], 1)
    return lmat


def TDMA(mat, values):
    mat_1 = np.zeros((3, len(mat)))
    for i in range(1, -2, -1):
        ls = i if i > 0 else None
        rs = i if i < 0 else None

        mat_1[1 - i, ls:rs] = np.diag(mat, i)

    # print(mat_1)
    sol = scipy.linalg.solve_banded((1, 1), mat_1, values)
    # print( sol, values)
    return sol


def solve(nt=10, ny=10, nx=10):
    at, bt = 0, 1
    ax, bx = 0, 1
    ay, by = 0, 1

    tl, ht = np.linspace(at, bt, nt + 1, retstep=True)
    xl, hx = np.linspace(ax, bx, nx + 1, retstep=True)
    yl, hy = np.linspace(ay, by, ny + 1, retstep=True)

    u = np.zeros((nt + 1, ny + 1, nx + 1))

    # u[0] = exact(*np.meshgrid(xl, yl), 0)
    for ti in range(nt+1):
        u[ti] = exact(*np.meshgrid(xl, yl), tl[ti])

    l1 = -l(nx) / hx ** 2
    l2 = -l(ny) / hy ** 2

    I1 = np.identity(len(l1))
    I2 = np.identity(len(l2))

    mat1 = (I1 + ht / 2 * l1)
    mat2 = (I2 + ht / 2 * l2)

    mat1[[0, -1]] = 0
    mat1[0, 0] = mat1[-1, -1] = 1
    mat2[[0, -1]] = 0
    mat2[0, 0] = mat2[-1, -1] = 1

    mat1f = (I1 - ht / 2 * l1)
    mat2f = (I2 - ht / 2 * l2)

    for ti in range(1, nt + 1):
        print(f"\r{ti}", end="")
        t2 = (tl[ti] + tl[ti-1])/2
        htf = ht/2 * f(*np.meshgrid(xl, yl), t2)

        tmp = mat2f @ u[ti - 1]
        tmp += htf
        tmp2 = np.zeros_like(tmp)
        for yi in range(ny+1):
            for j in [0, -1]:
                tmp[yi,j] = exact(xl[j], yl[yi], t2)
            tmp2[yi] = TDMA(mat1, tmp[yi])

        tmp = mat1f @ tmp2.T
        tmp = tmp.T + htf
        tmp2 = np.zeros_like(tmp)
        for xi in range(nx+1):
            for j in [0, -1]:
                tmp[j, xi] = exact(xl[xi], yl[j], tl[ti])
            tmp2[:, xi] = TDMA(mat2, tmp[:, xi])

        u[ti,1:-1,1:-1] = tmp2.copy()[1:-1,1:-1]


    return xl, yl, tl, u



from matplotlib.animation import FuncAnimation


def plot_3d():
    nt = 10000
    n = 10
    x, y, t, u = solve(nt, n, n)

    xx, yy = np.meshgrid(x, y)
    ue = np.zeros_like(u)
    for ti in range(nt + 1):
        ue[ti] = exact(xx, yy, t[ti])

    ht = 1 / nt
    zlim = [np.min([u, ue]) - 2 * ht, np.max([u, ue]) + 2 * ht]
    # print(zlim)

    def plot3d(ax, solution, ti):
        return ax.plot_surface(xx, yy, solution[ti],
                               cmap='plasma',
                               linewidth=0,
                               antialiased=False)

    fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})

    def update(ti):
        ax[0].clear()
        ax[1].clear()
        fig.suptitle(f"t = {np.round(t[ti], 2)}")
        ax[0].set_title("Численное")
        ax[1].set_title("Точное")
        ax[0].axes.set_zlim3d(bottom=zlim[0], top=zlim[1])
        ax[1].axes.set_zlim3d(bottom=zlim[0], top=zlim[1])
        plot3d(ax[0], u, ti)
        plot3d(ax[1], ue, ti)
        return

    update(0)

    frames, interval = range(0, nt + 1, nt // 10), 1000 / 10
    # print(nt + 1, nt // 10, frames)
    # print(f"ms: {interval}")

    ani = FuncAnimation(fig,
                        update,
                        frames=frames,
                        blit=False,
                        interval=interval)

    plt.show()

    diff = np.zeros(nt + 1)
    for ti in range(nt + 1):
        diff[ti] = np.max(abs(u[ti] - ue[ti])[:, :])
    # print(diff)

    plt.figure("Модуль разности")
    plt.plot(t, diff)

    plt.show()

def error_show():
    n = 100
    ntl = [10, 100, 1000]
    for nt in ntl:
        x, y, t, u = solve(nt, n, n)
        xx, yy = np.meshgrid(x, y)
        ue = np.zeros_like(u)
        for ti in range(len(u)):
            ue[ti] = exact(xx, yy, t[ti])

        diff = np.zeros(len(u))
        for ti in range(len(u)):
            diff[ti] = np.max(abs(u[ti] - ue[ti])[:, :])

        print("", np.max(diff))

# plot_3d()
error_show()