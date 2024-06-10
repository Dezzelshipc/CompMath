import numpy
import numpy as np
from numpy import pi as PI
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation
from scipy import interpolate
from multiprocessing import Pool

print(Warning("Вычисления могут быть очень долгими при больших делениях времени (nt) и количестве точек (cp)"))

def runge_kutta(function, y0: float | list, linsp):
    num = len(linsp)
    x_a = linsp
    h = linsp[1] - linsp[0]
    y_a = [y0] * num

    for i in range(num - 1):
        k0 = function(x_a[i], y_a[i])
        k1 = function(x_a[i] + h / 2, y_a[i] + h * k0 / 2)
        k2 = function(x_a[i] + h / 2, y_a[i] + h * k1 / 2)
        k3 = function(x_a[i] + h, y_a[i] + h * k2)
        y_a[i + 1] = y_a[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)

    return x_a, np.array(y_a)


def c0(x, y):
    return np.arctan((y - 0.5) / 0.1)


def u(x, y):
    return -PI * np.sin(2 * PI * x) * np.cos(PI * y)


def v(x, y):
    return 2 * PI * np.cos(2 * PI * x) * np.sin(PI * y)


def right(t, x):
    rnd = 0
    return np.array([
        u(x[0], x[1]),
        v(x[0], x[1])
    ]) + np.random.uniform(-rnd, rnd, 2)


def thread(point_i, points, tl):
    _, xy = runge_kutta(right, points[point_i], tl)
    return xy


def calc_grid(c_ti, values, xi):
    print(f"start")
    r = interpolate.griddata(c_ti, values, xi, method='cubic')
    print(f"end")
    return r


def calc_grid_rbf(c_ti, values, xi):
    print(f"start")
    flat = numpy.vstack((xi[0].ravel(), xi[1].ravel())).T
    r = interpolate.RBFInterpolator(c_ti,
                                    values,
                                    kernel='linear',
                                    neighbors=100,
                                    smoothing=1e-4,
                                    degree=0
                                    )(flat)
    r = r.reshape(xi[0].shape)
    print(f"end")
    return r


if __name__ == "__main__":

    nt = 200
    tl, dt = np.linspace(0, 0.4, nt + 1, retstep=True)

    cp = 2000

    # points = np.random.rand(cp, 2)

    points = np.zeros((cp, 2))
    points[:-2,0] = np.linspace(0, 1, cp-2)
    points[:-2,1] = 0.5
    points[-1] = [0.,0.]
    points[-2] = [1.,1.]
    print(points)

    values = c0(points[:, 0], points[:, 1])

    c = np.zeros((nt + 1, cp, 2))

    with Pool() as p:
        rng = [
            (
                i,
                points,
                tl
            ) for i in range(cp)
        ]
        res = np.moveaxis(p.starmap(thread, rng), 0, 1)
        c[:] = res

    print("done calc c")

    show_step = max(nt // 20, 1)
    print(f"{show_step=}")
    show = 2

    if show == 1:
        grid_y, grid_x = np.mgrid[0:1:500j, 0:1:500j]
        with Pool() as p:
            to_calc_grid = [
                (
                    c[ti],
                    values,
                    (grid_x, grid_y)
                ) for ti in range(0, nt + 1, show_step)
            ]
            grids = p.starmap(calc_grid_rbf, to_calc_grid)

        # grids = []
        # for ti in range(0, nt + 1, show_step):
        #     print(f"\r{ti}", end="")
        #     grids.append(calc_grid_rbf(c[ti], values, (grid_x, grid_y)))

        print("done calc grids")

        fig, ax = plt.subplots()
        plt.imshow(grids[0], extent=(0, 1, 0, 1), origin='lower')

        ax.set_title("0")


        def update(ti):
            ax.clear()
            plt.imshow(grids[ti], extent=(0, 1, 0, 1), origin='lower')
            ax.set_title(f"Момент времени: {np.round(tl[ti * show_step], 3)}")

        def update_save(ti):
            update(ti)
            # plt.savefig(f"./p{ti}.pdf")

        plt.colorbar()
        ani = FuncAnimation(fig, update_save, frames=len(grids), repeat=False)
        # ani = FuncAnimation(fig, update, frames=len(grids), interval=100)
    elif show == 2:
        fig, ax = plt.subplots()
        cm = matplotlib.colormaps['plasma']

        scatters = c[0:nt + 1:show_step]

        plt.scatter(scatters[0, :, 0], scatters[0, :, 1], c=values)

        ax.set_title("0")


        def update(ti):
            ax.clear()
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            plt.scatter(scatters[ti, :, 0], scatters[ti, :, 1], c=values)
            ax.set_title(f"Момент времени: {np.round(tl[ti * show_step], 3)}")

        def update_save(ti):
            update(ti)
            plt.savefig(f"./line{ti}.pdf")

        plt.colorbar()

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        # ani = FuncAnimation(fig, update, frames=len(scatters), interval=10)
        ani = FuncAnimation(fig, update_save, frames=len(scatters), repeat=False)
    elif show == 3:
        grid_y, grid_x = np.mgrid[0:1:20j, 0:1:20j]
        plt.streamplot(grid_x, grid_y, u(grid_x, grid_y), v(grid_x, grid_y), broken_streamlines=False, density=0.5)
        # plt.savefig(f"./streamplot.pdf")
    elif show == 4:
        grid_y, grid_x = np.mgrid[0:1:21j, 0:1:21j]
        plt.quiver(grid_x, grid_y, u(grid_x, grid_y), v(grid_x, grid_y))
        # plt.savefig(f"./quiver.pdf")

    plt.tight_layout(pad=1.03)
    plt.show()
    plt.close()