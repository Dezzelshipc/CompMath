import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from multiprocessing import Pool


def runge_kutta(function, y0: float, a: float, b: float, h: float):
    num = math.ceil((b - a) / h)
    x_a = np.linspace(a, b, num=num, endpoint=False)
    y_a = [y0] * num

    for i in range(num - 1):
        k0 = function(x_a[i], y_a[i])
        k1 = function(x_a[i] + h / 2, y_a[i] + h * k0 / 2)
        k2 = function(x_a[i] + h / 2, y_a[i] + h * k1 / 2)
        k3 = function(x_a[i] + h, y_a[i] + h * k2)
        y_a[i + 1] = y_a[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)

    return x_a, np.array(y_a)


# 1
def right_lin(t, ab):
    return np.array([
        ab[1],
        -w ** 2 * ab[0]
    ])


# 2
def right_sin(t, ab):
    return np.array([
        ab[1],
        -w ** 2 * np.sin(ab[0])
    ])


# 3
def right_fric(t, ab):
    return np.array([
        ab[1],
        -k * ab[1] - w ** 2 * ab[0]
    ])


# 4
def right_force(t, ab):
    return np.array([
        ab[1],
        Af * np.sin(wf * t) - w ** 2 * ab[0]
    ])


# 5
def right_force_firc(t, ab):
    return np.array([
        ab[1],
        Af * np.sin(wf * t) - k * ab[1] - w ** 2 * ab[0]
    ])


def right_force_firc2(t, ab, Af, wf, k):
    return np.array([
        ab[1],
        Af * np.sin(wf * t) - k * ab[1] - w ** 2 * ab[0]
    ])


def model(y0, right):
    x_, y_ = runge_kutta(right, y0, t0, tn, (tn - t0) / n)
    y_ = y_.T

    return x_, y_


def max_amp(y0, n, t0, tn, Af, wf, k):
    rff = lambda t, x: right_force_firc2(t, x, Af, wf, k)
    _, y_ = runge_kutta(rff, y0, t0, tn, (tn - t0) / n)
    y_ = y_.T
    return max(abs(y_[0][-max(n // 4, 100): -1]))


def res_an(t, w, k, Af):
    s = k / (2 * w)
    return Af / np.sqrt((2 * w * s * t) ** 2 + (w ** 2 - t ** 2) ** 2)


t0, tn = 0, 15
n = 10000

g = 9.8
L = 1
w = np.sqrt(g / L)

k = 0.1
Af = 1
wf = w / 2
if __name__ == "__main__":

    init = [np.pi / 10, 0]
    asd1 = False
    # asd1 = True
    if asd1:
        x_, y_1 = model(init, right_lin)
        x_, y_2 = model(init, right_sin)
        # plt.figure(0)
        # plt.xlabel("t")
        # plt.ylabel("a")
        # plt.plot(x_, y_1[0])

        # plt.plot(x_, y_2[0])

        plt.figure(1)
        plt.xlabel("a")
        plt.ylabel("a'")
        plt.plot(y_1[0], y_1[1])
        plt.plot(y_2[0], y_2[1])

    asd2 = False
    # asd2 = True
    if asd2:
        tn = 50
        x_, y_ = model(init, right_fric)

        # plt.figure(5)
        # plt.xlabel("t")
        # plt.ylabel("a")

        # plt.plot(x_, y_[0])

        plt.figure(6)
        plt.xlabel("a")
        plt.ylabel("a'")

        plt.plot(y_[0], y_[1], marker='o', markevery=[0])

    asd3 = False
    # asd3 = True
    if asd3:
        tn = 25
        wf = w / 2
        x_, y_ = model(init, right_force)
        plt.figure(70)
        plt.xlabel("t")
        plt.ylabel("a")
        plt.plot(x_, y_[0])
        plt.plot(x_, Af * np.sin(wf * x_))

        # plt.figure(71)
        # plt.xlabel("a")
        # plt.ylabel("a'")
        # plt.plot(y_[0], y_[1], marker='o', markevery=[0])

    asd4 = False
    # asd4 = True
    if asd4:
        wf = 0.5
        ne = n // 4
        tn = 100
        n = 10000

        x_, y_ = model(init, right_force_firc)

        # plt.figure(90)
        # ax = plt.gca()
        # ax.set_prop_cycle('color',plt.cm.plasma(np.linspace(0,1,n)))

        # for i in range(n-1):
        #     plt.plot(x_[i:i+2], y_[0][i:i+2])

        # plt.figure(91)

        # ax = plt.gca()
        # ax.set_prop_cycle('color',plt.cm.plasma(np.linspace(0,1,n)))
        # for i in range(n-1):
        #     plt.plot(y_[0][i:i+2], y_[1][i:i+2])

        plt.plot(x_[:ne], y_[0][:ne])
        plt.plot(x_[ne:], y_[0][ne:])
        plt.xlabel("t")
        plt.ylabel("a")

        # plt.plot(y_[0][:ne], y_[1][:ne], marker='o', markevery=[0])
        # plt.plot(y_[0][ne:], y_[1][ne:])
        # plt.xlabel("a")
        # plt.ylabel("a'")

    # 5
    asd5 = False
    asd5 = True
    if asd5:
        plt.figure(100)

        t0, tn = 0, 500
        n = 10000

        c = 60
        wa, wb = 0.5, 2
        wfs = np.linspace(wa, wb, c + 1)
        analitic_w = np.linspace(wa, wb, 10000)

        wfs = w * wfs
        ks = [1, 0.75, 0.5]
        colors = ['c', 'b', 'y', 'r', 'm', 'g']
        max_line = []
        max_line_a = []
        for i, k in enumerate(ks):
            print(i, k)
            args = [
                (init, n, t0, tn, Af, wfi, k) for wfi in wfs
            ]
            with Pool() as p:
                maxes = p.starmap(max_amp, args)

            res_analityc = res_an(analitic_w * w, w, k, Af)
            plt.plot(analitic_w, res_analityc, c=colors[2*i])
            plt.plot(wfs / w, maxes, 'o', markersize=3, c=colors[2*i+1])

            argmax = np.argmax(maxes)
            max_line.append( (wfs[argmax] / w, maxes[argmax]) )

            argmax_a = np.argmax(res_analityc)
            max_line_a.append( (analitic_w[argmax_a], res_analityc[argmax_a]) )

        plt.xlabel("w")
        plt.ylabel("A")

        max_line = np.array(max_line).T
        plt.plot(max_line[0], max_line[1], '--')
        max_line_a = np.array(max_line_a).T
        plt.plot(max_line_a[0], max_line_a[1], '--')

        plt.title("Зависимость амплитуды от частоты вынуждающих колебаний")
        legend = np.array([(f"Аналитическое k = {ks[i]}", f"Численное k = {ks[i]}") for i in range(len(ks))])
        plt.legend(np.append(legend.flatten(), ["Числ. линия макс. амплитуд", "Аналит. линия макс. амплитуд"]))

    # plt.savefig("./sem6-matmodelling/3 pendulum/figure.pdf")
    plt.grid()
    plt.tight_layout(pad=1.03)
    plt.show()