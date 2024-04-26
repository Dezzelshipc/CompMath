import numpy as np
import math
import matplotlib.pyplot as plt


ksi1,ksi2,ksi3 = 1,2,3
v12,v13,v23=1,2,3
k12,k13,k23 = 1,2,3

def ody(t,x):
    x_1 = (ksi1 - v12*x[1] - v13*x[2]) *x[0]
    x_2 = (ksi2 + k12*v12*x[0] - v23*x[2])*x[1]
    x_3 = (-ksi3 + k13*v13*x[0] + k23*v23*x[1])*x[2]
    return np.array([x_1, x_2, x_3])

def right(t, x):
    return np.array([
        ( ksi1 - v12 * x[1] - v13 * x[2] ) * x[0],
        ( ksi2 + k12 * v12 * x[0] - v13 * x[2] ) * x[1],
        (-ksi3 + k13 * v13 * x[0] + k23 * v23 * x[1] ) * x[2]
    ])
    
ody = right

def RungeKutta(y,h,x,N):
    for i in range(N-1):
        k0 = ody(x[i], y[i])
        k1 = ody(x[i] + h / 2, y[i] + h * k0 / 2)
        k2 = ody(x[i] + h / 2, y[i] + h * k1 / 2)
        k3 = ody(x[i] + h, y[i] + h * k2)

        y[i+1] = y[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)
    return y

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


a,b = 0,100
N = 1000
h = (b-a)/N
X = np.linspace(a,b,N,endpoint=False)
y = np.zeros((N,3 ))
y[0] = [1,1,1]
X, r = runge_kutta(ody, [1,1,1], a, b, h)
# r = RungeKutta(y, h, X, N)

r = r.T

plt.plot(X,r[0])
plt.plot(X,r[1])
plt.plot(X,r[2])
plt.legend(['Жертва_1', 'Жертва_2', 'Хищник'])
plt.show()