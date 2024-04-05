import numpy as np
import scipy as sci 
import matplotlib.pyplot as plt 
import math


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


def count_time(a, b, c, d, y0):
    plt.figure("C(t)")
    x, y = model(a, b, c, d, np.array(y0))

    plt.plot(x, y[0])
    plt.plot(x, y[1])
    
    plt.xlabel('t - Время')
    plt.ylabel('C(t) - Количество')
    plt.legend(["N - жертвы", "M - хищники"])
    
    plt.xlim(left=0)
    plt.ylim(bottom=0)


def phase(a, b, c, d, init):
    plt.figure("Phase")
    for i in init:
        x, y = model(a, b, c, d, np.array(i))
        plt.plot(y[0], y[1], marker='o', markevery=[0])
    
    plt.plot(b/d, a/c, 'ro')
    plt.legend(init)
    plt.xlabel('N - жертвы')
    plt.ylabel('M - хищники')
    
    plt.xlim(left=0)
    plt.ylim(bottom=0)

def model(a, b, c, d, y0):
    def diff(t, NM):
        return np.array([
            (a - c * NM[1]) * NM[0],
            (-b + d * NM[0]) * NM[1]
        ])


    x_ = np.linspace(t0, tn, n)
    x_, y_ = runge_kutta(diff, y0, t0, tn, (tn-t0)/n)
    y_ = y_.T
    
    return x_, y_

    
    
    # plt.plot(y[1], y[0])

t0, tn = 0, 10
n = 1000

a, c = 2, 0.5
b, d = 1, 0.5

init_val = [[4,4], [2,6], [3,2], [5, 3], [2,3], [2,1]]

# count_time(a,b,c,d, [4,4])

phase(a,b,c,d, init_val)

# plt.savefig("./sem6-matmodelling/asd.png")

plt.show()