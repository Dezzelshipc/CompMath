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

# 1
def right_lin(t, ab):
    return np.array([
        ab[1],
        -w**2 * ab[0]
    ])
 
# 2   
def right_sin(t, ab):
    return np.array([
        ab[1],
        -w**2 * np.sin( ab[0] )
    ])

# 3
def right_fric(t, ab):
    return np.array([
        ab[1],
        -k * ab[1] - w**2 * ab[0]
    ])

# 4
def right_force(t, ab):
    return np.array([
        ab[1],
        Af * np.sin(wf * t) - w**2 * ab[0]
    ])

# 5
def right_force_firc(t, ab):
    return np.array([
        ab[1],
        Af * np.sin(wf * t) - k * ab[1] - w**2 * ab[0]
    ])

def model(y0, right):
    x_, y_ = runge_kutta(right, y0, t0, tn, (tn-t0)/n)
    y_ = y_.T
    
    return x_, y_

t0, tn = 0, 150
n = 1000

g = 9.8
L = 1
w = np.sqrt( g / L )

k = 0.1
Af = 2
wf = 0.5

init = [np.pi/10, 0]
# x_, y_ = model(init, right_lin)
# plt.plot(x_, y_[0])

# x_, y_ = model(init, right_sin)
# plt.plot(x_, y_[0])

# x_, y_ = model(init, right_fric)
# plt.plot(x_, y_[0])

# x_, y_ = model(init, right_force)
# plt.plot(x_, y_[0])


# 5
c = 100
hw = 0.005
wl = []
Al = []
for i in range(-c, c+1):
    wf = w + hw * i
    x_, y_ = model(init, right_force_firc)
    wl.append( wf )
    Al.append(max(abs( y_[0][-n//4 : -1] )))
    
plt.plot(wl, Al)
plt.xlabel("w")
plt.ylabel("A")



plt.show()