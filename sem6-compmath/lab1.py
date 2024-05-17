# some methods
import numpy as np
import math
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def euler(function, y0: float, a: float, b: float, h: float):
    num = math.ceil((b - a) / h )
    x_a = np.linspace(a, b, num=num, endpoint=False)
    
    y_a = [y0] * num

    for i in range(num - 1):
        y_a[i + 1] = y_a[i] + h * function(x_a[i], y_a[i])

    return x_a, np.array(y_a)


def euler_recalc(function, y0: float, a: float, b: float, h: float):
    num = math.ceil((b - a) / h )
    x_a = np.linspace(a, b, num=num, endpoint=False)
    y_a = [y0] * num

    for i in range(num - 1):
        y_r = y_a[i] + h * function(x_a[i], y_a[i])
        y_a[i + 1] = y_a[i] + h / 2 * (function(x_a[i], y_a[i]) + function(x_a[i+1], y_r))

    return x_a, np.array(y_a)


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


def plot_funcs(f, y0: float, a: float, b: float, h: float):
    x, y_e = euler(f[0], y0, a, b, h)
    _, y_er = euler_recalc(f[0], y0, a, b, h)
    _, y_rk = runge_kutta(f[0], y0, a, b, h)

    y_f = f[1](x)

    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.subplots_adjust(bottom=0.25)
    x_f = np.linspace(a, b, num=10000)

    ax1.plot(x_f, f[1](x_f), 'g')
    e_plot, = ax1.plot(x, y_e, 'b')
    er_plot, = ax1.plot(x, y_er, 'c')
    rk_plot, = ax1.plot(x, y_rk, 'r')
    
    e_d = np.abs(y_f - y_e)
    er_d =  np.abs(y_f - y_er)
    rk_d = np.abs(y_f - y_rk)

    e_d_plot, = ax2.plot(x, e_d, 'b')
    er_d_plot, = ax2.plot(x, er_d, 'c')
    rk_d_plot, = ax2.plot(x, rk_d, 'r')

    ax1.set_title("Functions")
    ax2.set_title("Error")

    ax1.set_xlabel("x")
    ax1.set_ylabel("y")

    ax2.set_xlabel(f"{h=:.5f},\nn={int((b-a)/h)+1}")

    ax1.legend(['Func', 'Euler', 'Euler Recalc', 'Runge-Kutta'])
    ax2.legend([max(e_d), max(er_d), max(rk_d)])
    
    h_ax = fig.add_axes([0.1, 0.1, 0.8, 0.03])
    h_slider = Slider(
        ax=h_ax,
        label='log2 h',
        valmin=-12,
        valmax=1,
        valinit=int(np.log2(h)),
    )
    
    def upd(val_h):
        h = 2 ** val_h
        ax2.set_xlabel(f"{h=:.5f},\nn={int((b-a)/h)+1}")

        x, y_e = euler(f[0], y0, a, b, h)
        _, y_er = euler_recalc(f[0], y0, a, b, h)
        _, y_rk = runge_kutta(f[0], y0, a, b, h)

        y_f = f[1](x)
        
        e_plot.set_data(x, y_e)
        er_plot.set_data(x, y_er)
        rk_plot.set_data(x, y_rk)
        
        e_d = np.abs(y_f - y_e)
        er_d =  np.abs(y_f - y_er)
        rk_d = np.abs(y_f - y_rk)
        
        e_d_plot.set_data(x, e_d)
        er_d_plot.set_data(x, er_d)
        rk_d_plot.set_data(x, rk_d)
        
        ax2.legend([max(e_d), max(er_d), max(rk_d)])
        
        ax2.relim()
        ax2.autoscale_view()
        
        fig.canvas.draw_idle()
    
    h_slider.on_changed(upd)

    # fig.canvas.draw_idle()
    
    return h_slider


def plot_funcs_h(f, y0: float, a: float, b: float, h: float):
    n = int(-np.log2(h))
    h_a = np.logspace(-1, -n, base=2, num=n)

    y_e_e = [0] * n
    y_er_e = [0] * n
    y_rk_e = [0] * n

    for i in range(n):
        hi = h_a[i]

        x, y_e = euler(f[0], y0, a, b, hi)
        _, y_er = euler_recalc(f[0], y0, a, b, hi)
        _, y_rk = runge_kutta(f[0], y0, a, b, hi)
        y_f = f[1](x)

        y_e_e[i] = np.max(np.abs(y_f - y_e))
        y_er_e[i] = np.max(np.abs(y_f - y_er))
        y_rk_e[i] = np.max(np.abs(y_f - y_rk))

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

    ax1.plot(h_a, y_e_e, 'bo-')
    ax1.plot(h_a, y_er_e, 'co-')
    ax1.plot(h_a, y_rk_e, 'ro-')

    ax1.set_title("Errors Log")
    ax2.set_title("Errors Log x, norm y")
    ax3.set_title("Errors Normal")

    ax1.legend(['Euler', 'Euler Recalc', 'Runge-Kutta'])
    ax1.set_xlabel("h")
    ax1.set_ylabel("max abs diff")

    ax1.set_xscale('log')
    ax1.set_yscale('log')
    
    ax2.plot(h_a, y_e_e, 'bo-')
    ax2.plot(h_a, y_er_e, 'co-')
    ax2.plot(h_a, y_rk_e, 'ro-')
    
    ax2.set_xscale('log')

    ax3.plot(h_a, y_e_e, 'bo-')
    ax3.plot(h_a, y_er_e, 'co-')
    ax3.plot(h_a, y_rk_e, 'ro-')
    
    print("h, e, er, rk")
    print(*(zip(h_a, y_e_e, y_er_e, y_rk_e)), sep="\n")
    
    
    # fig.canvas.draw_idle()
    

class Test(Enum):
    NONE = -1
    ONE = 0
    X2 = 1
    SINX = 2
    SQRT = 3
    XY = 4
    COS2Y = 5
    LN = 6
    SINX2Y = 7
    ESINYCOS = 8
    


if __name__ == '__main__':

    y0 = 0
    x0, xn = 0, 1
    h = 0.1

    t = Test.ESINYCOS

    match t:
        case Test.ONE:
            f = (
                lambda a, b: 1,
                lambda a: a + (y0 - x0)
            )

        case Test.X2:
            f = (
                lambda a, b: 2 * a,
                lambda a: a ** 2 + (y0 - x0 ** 2)
            )

        case Test.SINX:
            f = (
                lambda a, b: np.sin(a),
                lambda a: -np.cos(a) + (y0 + np.cos(x0))
            )

        case Test.SQRT:
            f = (
                lambda a, b: np.sqrt(a),
                lambda a: a ** 1.5 * 2 / 3 + (y0 - 2 / 3 * x0 ** 1.5)
            )

        case Test.XY:
            f = (
                lambda a, b: a + b,
                lambda a: (y0 + x0 + 1) * np.e ** (a - x0) - a - 1
            )

        case Test.COS2Y:
            f = (
                lambda a, b: np.cos(b) ** 2,
                lambda a: np.arctan(np.tan(y0) - x0 + a)
            )

        case Test.LN:
            f = (
                lambda a, b: 1 / a,
                lambda a: np.log(a) + (y0 + np.log(x0))
            )

        case Test.SINX2Y:
            sol = lambda x: np.e ** (0.5 * (x - np.sin(x) * np.cos(x)))

            f = (
                lambda a, b: (np.sin(a) ** 2) * b,
                lambda a: y0 / sol(x0) * sol(a)
            )

        case Test.ESINYCOS:
            
            y0 = 0
            x0 = 0
            
            f = (
                lambda a, b: np.exp(-np.sin(a)) - b * np.cos(a),
                lambda a: a * np.exp(-np.sin(a))
            )

    sliders = plot_funcs(f, y0, x0, xn, h)

    plot_funcs_h(f, y0, x0, xn, h)

    plt.autoscale()
    plt.show()
