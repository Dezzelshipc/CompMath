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


KC = 276
sigma = 5.67e-8

T_l = 190 + KC
T_u = 200 + KC


is_turned = True
def H(T):
    global is_turned

    if T > T_u:
        is_turned = False
    elif T < T_l:
        is_turned = True
    
    return int(is_turned)

def H0(T):
    return 1.

def H1(T):
    return 1 if T < T_l else (T_u - T)/(T_u - T_l)


def utug(P, m, c, S, k):
    def dTdt(t, T):
        return (P * H(T) - k * S * (T - T0) - sigma * S * (T**4 - T0**4)) / (c * m)
    
    x = np.linspace(a, b, n)

    x, y = runge_kutta(dTdt, T0, a, b, (b-a)/n)
    y -= KC
    plt.plot(x, y)

a, b = 0, 100
n = 10000


P = 3000
m = 0.5
c = 897  # Алюминий
S = 0.4
k = 10
T0 = 20 + KC


utug(P, m, c, S, k)
    
# plt.legend(ms)
plt.xlabel('t - Время')
plt.ylabel('T(t) - Температура в цельсиях')

plt.xlim([a,b])
# plt.ylim([min(y_), max(y_)+ 10])

plt.savefig("./sem6-matmodelling/asd.png")

plt.show()