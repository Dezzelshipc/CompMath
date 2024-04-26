import numpy as np
import math
import matplotlib.pyplot as plt


def runge_kutta(function, y0: float, a: float, b: float, h: float):
    num = math.ceil((b - a) / h)
    x_a = np.linspace(a, b, num=num, endpoint=False)
    y_a = [y0] * (num)

    for i in range(num - 1):
        k0 = function(x_a[i], y_a[i])
        k1 = function(x_a[i] + h / 2, y_a[i] + h * k0 / 2)
        k2 = function(x_a[i] + h / 2, y_a[i] + h * k1 / 2)
        k3 = function(x_a[i] + h, y_a[i] + h * k2)
        y_a[i + 1] = y_a[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)

    return x_a, np.array(y_a)


ksi1, ksi2, ksi3 = 1, 2, 3
a12, a13, a23 = 1, 2, 3
k12, k13, k23 = 1, 2, 3


def static_point():
    ld = k13 - k12 * k23
    if ld != 0:
        return np.array([
            (ksi3 * a12 - ksi1 * a23 * k23 + ksi2 * k23 * a13) / (a12 * a13 * ld),
            (ksi1 * a13 * k13 - ksi2 * a13 * k13 - ksi3 * a12 * k12) / (a12 * a23 * ld),
            (ksi3 * a12 * k12 + ksi2 * a13 * k13 - ksi1 * k23 * k12 * k23) / (a13 * a23 * ld),
        ])
    else:
        x3 = 1
        return np.array([
            (a23 * x3 - ksi2) / (a12 * k12),
            (-a13 * x3 + ksi1) / a12,
            x3,              
        ])

def right(t, x):
    return np.array([
        ( ksi1 - a12 * x[1] - a13 * x[2] ) * x[0],
        ( ksi2 + k12 * a12 * x[0] - a13 * x[2] ) * x[1],
        (-ksi3 + k13 * a13 * x[0] + k23 * a23 * x[1] ) * x[2]
    ])
    
def ody(t,x):
    x_1 = (ksi1 - a12*x[1] - a13*x[2]) *x[0]
    x_2 = (ksi2 + k12*a12*x[0] - a23*x[2])*x[1]
    x_3 = (-ksi3 + k13*a13*x[0] + k23*a23*x[1])*x[2]
    return np.array([x_1, x_2, x_3])


def plot3(tl, xl):
    plt.figure(0)
    plt.plot(tl, xl[0])
    plt.plot(tl, xl[1])
    plt.plot(tl, xl[2])

    plt.legend(["x1", "x2", "x3"])
    
def plotp(tl, xl, n1, n2):
    plt.figure(f"{n1}{n2}")
    plt.plot(xl[n1], xl[n2])
    plt.xlabel(f"x{n1+1}")
    plt.ylabel(f"x{n2+1}")
    
    
def plotp3(tl, xl):
    ax = plt.figure("123").add_subplot(projection='3d')
    
    ax.plot(xl[0], xl[1], xl[2])
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')

a, b = 0, 10
n = 10000
h = (b-a)/n

x0 = np.array([1, 1, 1])
# x0 = static_point()
tl, xl = runge_kutta(ody, x0, a, b, h)
xl = xl.T

print(x0)

plot3(tl, xl)

plotp(tl, xl, 0, 1)
plotp(tl, xl, 0, 2)
plotp(tl, xl, 1, 2)

plotp3(tl, xl)

a2 = k12 * a12**2 * x0[0] * x0[1] + k13 * a13**2 * x0[0]*x0[2] + k23 * a23 ** 2 * x0[1] * x0[2]
print(f"{a2=}")
a3 = x0[0] * x0[1] * x0[2] * a12 * a13* a23 * (k13- k12 * k23)
print(f"{a3=}")


plt.show()