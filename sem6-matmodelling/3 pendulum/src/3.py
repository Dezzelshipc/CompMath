import numpy as np
import matplotlib.pyplot as plt 
import matplotlib.cm as cm
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

t0, tn = 0, 15
n = 10000

g = 9.8
L = 1
w = np.sqrt( g / L )

k = 0.1
Af = 1
wf = 0.5

init = [np.pi/100, 0]
asd1 = False
# asd1 = True
if asd1:
    plt.figure(0)
    x_, y_1 = model(init, right_lin)
    plt.plot(x_, y_1[0])

    x_, y_2 = model(init, right_sin)
    plt.plot(x_, y_2[0])

    plt.figure(1)
    plt.plot(y_1[0], y_1[1])
    plt.plot(y_2[0], y_2[1])

asd2 = False
# asd2 = True
if asd2:
    plt.figure(5)
    tn = 50
    x_, y_ = model(init, right_fric)
    plt.plot(x_, y_[0])

    plt.figure(6)
    plt.plot(y_[0], y_[1])

asd3 = False
# asd3 = True
if asd3:
    plt.figure(70)
    x_, y_ = model(init, right_force)
    plt.plot(x_, y_[0])

    plt.figure(71)
    plt.plot(y_[0], y_[1])


asd4 = False
# asd4 = True
if asd4:
    wf = w
    ne = 3* n//4
    tn = 150

    x_, y_ = model(init, right_force_firc)

    plt.figure(90)
    ax = plt.gca()
    ax.set_prop_cycle('color',plt.cm.plasma(np.linspace(0,1,n)))
    
    for i in range(n-1):
        plt.plot(x_[i:i+2], y_[0][i:i+2])
        
    plt.figure(91)

    ax = plt.gca()
    ax.set_prop_cycle('color',plt.cm.plasma(np.linspace(0,1,n)))
    for i in range(n-1):
        plt.plot(y_[0][i:i+2], y_[1][i:i+2])

# 5
asd5 = False
asd5 = True
if asd5:
    plt.figure(100)

    t0, tn = 0, 150
    n = 1000

    c = 121
    wh = 0.2
    wfs = np.linspace(w-wh, w+wh, c+1)
    
    Al = []
    for wfi in wfs:
        wf = wfi
        x_, y_ = model(init, right_force_firc)
        Al.append(max(abs( y_[0][-n//4 : -1] )))
    

    plt.plot(wfs, Al, 'o-', markersize=3)
    plt.xlabel("w")
    plt.ylabel("A")

    Al = np.array(Al)

    plt.plot([w,w], [min(Al), max(Al)], '--')
    wm = wfs[Al.argmax()]
    plt.plot([wm, wm], [min(Al), max(Al)], '--')

    plt.title("Зависимость амплитуды от частоты вынуждающих колебаний")
    plt.legend(["Кривая амплитуд", f"Собственная частота ({w})", f"Резонансная частота ({wm})"])

plt.show()