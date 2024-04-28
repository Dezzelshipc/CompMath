import numpy as np 
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

t0, tn = 0, 10
n = 1000

a, c = 2, 2
b, d = 1, 4

t, x = model(a,b,c,d, [2,2])