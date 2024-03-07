import numpy as np
import scipy as sci 
import matplotlib.pyplot as plt 
import math


def runge_kutta(function, y0: float, a: float, b: float, h: float):
    num = math.ceil((b - a) / h + 1)
    x_a = np.linspace(a, b, num=num)
    y_a = [y0] * num

    for i in range(num - 1):
        k0 = function(x_a[i], y_a[i])
        k1 = function(x_a[i] + h / 2, y_a[i] + h * k0 / 2)
        k2 = function(x_a[i] + h / 2, y_a[i] + h * k1 / 2)
        k3 = function(x_a[i] + h, y_a[i] + h * k2)
        y_a[i + 1] = y_a[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)

    return x_a, np.array(y_a)


KC = 276

P = 2000
m = 0.5
c = 920
S = 0.4
k = 10
T0 = 300
sigma = 1.380649e-23


T_l = 290 + KC
T_u = 300 + KC

is_turned = True
def H(T):
    global is_turned

    if is_turned and T > T_u:
        is_turned = False
    elif not is_turned and T < T_l:
        is_turned = True
    
    return float(is_turned)
        

def dTdt(t, T):
    return (P * H(T) - k * S * (T - T0) - sigma * S * (T**4 - T0**4)) / (c * m)

def dTdt2(T, t):
    return dTdt(t, T)

a, b = 0, 500
n = 10000

x = np.linspace(a, b, n)
# y = sci.integrate.odeint(dTdt2, T0, x)
y = (sci.integrate.solve_ivp(dTdt, [a, b], [T0], t_eval=x)).y[0]
y = np.array(y)

x_, y_ = runge_kutta(dTdt, T0, a, b, (b-a)/n)


# plt.plot(x, y - KC)
plt.plot(x_, y_ - KC)
plt.show()