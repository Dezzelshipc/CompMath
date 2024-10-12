# 2-cycle splitting for parabolic pde (heat eq)
import numpy as np
import matplotlib.pyplot as plt


def exact(x, y, t):
    return t * np.exp(x+y)
    # return t * np.sin(np.pi * x) * np.sin(np.pi * y)
    # return t + (y ** 2 + x ** 2)
    # return t + (y ** 2 + x ** 2) / 4


def f(x, y, t):
    return (1 - 2 * t) * np.exp(x+y)
    # return (1 + 2 * t * np.pi ** 2) * np.sin(np.pi * x) * np.sin(np.pi * y)
    # return -3
    # return 0


def l(size):
    lmat = -2 * np.identity(size + 1)
    ones1 = np.ones(size + 1)
    ones1[0] = ones1[-1] = 0
    lmat += np.diag(ones1[1:], -1) + np.diag(ones1[:-1], 1)
    lmat[0, 0] = lmat[-1, -1] = 1
    # lmat[0, :3] = [1, -2, 1]
    # lmat[-1, -3:] = [1, -2, 1]
    return lmat


at, bt = 0, 1
ax, bx = 0, 1
ay, by = 0, 1

nt = 20
nx = 20
ny = 20

tl, ht = np.linspace(at, bt, nt + 1, retstep=True)
xl, hx = np.linspace(ax, bx, nx + 1, retstep=True)
yl, hy = np.linspace(ay, by, ny + 1, retstep=True)

u = np.zeros((nt + 1, ny + 1, nx + 1))

xx, yy = np.meshgrid(xl, yl)

u[0] = exact(*np.meshgrid(xl, yl), 0)

l1 = -l(nx) / hx ** 2
l2 = -l(ny) / hy ** 2


h_step = np.zeros((ny + 1, nx + 1))
for ti in range(1, nt + 1, 2):
    print(f"\r{ti}", end="")
    def half_step(lmat: np.matrix, vals: np.array, ex0, ex1):
        I = np.identity(len(lmat))
        q_step = (I - ht / 2 * lmat) @ vals
        q_step[0] = ex0
        q_step[-1] = ex1

        mat = I + ht / 2 * lmat
        mat[0] = mat[-1] = 0
        mat[0, 0] = mat[-1, -1] = 1
        return np.linalg.solve(mat, q_step)


    for yi in range(ny + 1):
        e0, e1 = exact(np.array([xl[0], xl[-1]]), yl[yi], (tl[ti-1] + tl[ti]) / 2)
        h_step[yi] = half_step(l1, u[ti - 1, yi], e0, e1)
    u[ti] = h_step.copy()

    for xi in range(nx + 1):
        e0, e1 = exact(xl[xi], np.array([yl[0], yl[-1]]), tl[ti])
        h_step[:, xi] = half_step(l2, u[ti, :, xi], e0, e1)
    u[ti] = (h_step.copy() + ht * f(*np.meshgrid(xl, yl), tl[ti]))


    for xi in range(nx + 1):
        e0, e1 = exact(xl[xi], np.array([yl[0], yl[-1]]), (tl[ti] + tl[ti+1]) / 2)
        h_step[:, xi] = half_step(l2, u[ti, :, xi] + ht * f(xl[xi], yl, tl[ti]), e0, e1)
    u[ti + 1] = h_step.copy()

    for yi in range(ny + 1):
        e0, e1 = exact(np.array([xl[0], xl[-1]]), yl[yi], tl[ti+1])
        h_step[yi] = half_step(l1, u[ti + 1, yi], e0, e1)
    u[ti + 1] = h_step.copy()



ue = u.copy()
for ti in range(1, nt + 1):
    ue[ti] = exact(xx, yy, tl[ti])

from matplotlib.animation import FuncAnimation
from matplotlib import cm

xx, yy = np.meshgrid(xl, yl)
zlim = [np.min([u, ue]) - 2 * ht, np.max([u, ue]) + 2 * ht]
print(zlim)


def plot3d(ax, solution, ti):
    return ax.plot_surface(xx, yy, solution[ti],
                           cmap=cm.plasma,
                           linewidth=0,
                           antialiased=False)


fig, ax = plt.subplots(1, 2, subplot_kw={"projection": "3d"})


def update(ti):
    ax[0].clear()
    ax[1].clear()
    fig.suptitle(f"t = {np.round(tl[ti], 2)}")
    ax[0].set_title("Численное")
    ax[1].set_title("Точное")
    ax[0].axes.set_zlim3d(bottom=zlim[0], top=zlim[1])
    ax[1].axes.set_zlim3d(bottom=zlim[0], top=zlim[1])
    plot3d(ax[0], u, ti)
    plot3d(ax[1], ue, ti)
    return


update(0)

frames, interval = range(nt + 1), 1000 * (bt - at) / nt
print(f"ms: {interval}")

ani = FuncAnimation(fig,
                    update,
                    frames=frames,
                    blit=False,
                    interval=interval)

diff = np.zeros(nt + 1)
for ti in range(nt + 1):
    diff[ti] = np.max((u[ti] - ue[ti]))
# print(diff)

plt.figure("Модуль разности")
plt.plot(tl, diff)

plt.show()
