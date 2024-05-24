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
        x[2], # u
        x[3], # v
        2 * w * x[3], # du
        -2* w * x[2], # dv
    ])


t0, tn = 0, 10
n = 1000
    
w = 1

def model(init):
    t, x = runge_kutta(right, init, t0, tn, (tn-t0)/n)
    x = x.T

    plt.plot(x[0] , x[1], marker='o', markevery=[0])

model([5,3,4,4])
model([5,3,5,5])
model([5,3,1,1])
model([5,3,-1,-1])

plt.grid(True)
plt.axhline(y=0, color='k')
# plt.axvline(x=0, color='k')
plt.show()