import numpy as np
import matplotlib.pyplot as plt


def runge_kutta(function, y0: float, a: float, b: float, h: float):
    num = int(np.ceil((b - a) / h))
    x_a = np.linspace(a, b, num=num, endpoint=False)
    y_a = [y0] * num

    for i in range(num - 1):
        k0 = function(x_a[i], y_a[i])
        k1 = function(x_a[i] + h / 2, y_a[i] + h * k0 / 2)
        k2 = function(x_a[i] + h / 2, y_a[i] + h * k1 / 2)
        k3 = function(x_a[i] + h, y_a[i] + h * k2)
        y_a[i + 1] = y_a[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)

    return x_a, np.array(y_a)


def right(t, x):
    return np.array([
        x[2], # u, dx
        x[3], # v, dy
        2 * w * x[3], # du, ddx
        -2* w * x[2], # dv, ddy
    ])


t0, tn = 0, 5
n = 100000


def model(inits):
    conserv = []
    for x0 in inits:
        t, x = runge_kutta(right, x0, t0, tn, (tn-t0)/n)
        x = x.T

        conv = ( x[2]**2 + x[3]**2 - x0[2]**2 - x0[3]**2 )/(x0[2]**2 + x0[3]**2)
        conserv.append(conv)

        plt.plot(x[0] , x[1], marker='o', markevery=[0])

    plt.grid(True)
    plt.axhline(y=0, color='k')
    plt.legend(list(map(list, np.round(inits, 3))), loc='upper right')
    ax = plt.gca()
    ax.set_aspect('equal', adjustable='datalim')


    plt.figure("convers")
    for c in conserv:
        plt.plot(t, c)
        
    plt.legend(list(map(list, np.round(inits, 3))), loc='upper right')
    plt.grid(True)
    plt.axhline(y=0, color='k')


w = 1
x0s = [
    [5,3,0,6],
    [5,3,5,5],
    [5,3,1,1],
    [5,3,-3,3],
    [5,3,-4,-2],
    [5,3,1,-3]
]



model(x0s)

# plt.axvline(x=0, color='k')
plt.show()