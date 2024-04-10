import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def TDMA(a,b,c,f):
    a, b, c, f = tuple(map(lambda k_list: list(map(float, k_list)), (a, c, b, f)))

    alpha = [-b[0] / c[0]]
    beta = [f[0] / c[0]]
    n = len(f)
    x = [0]*n

    for i in range(1, n):
        alpha.append(-b[i]/(a[i]*alpha[i-1] + c[i]))
        beta.append((f[i] - a[i]*beta[i-1])/(a[i]*alpha[i-1] + c[i]))

    x[n-1] = beta[n - 1]

    for i in range(n-1, -1, -1):
        x[i - 1] = alpha[i - 1]*x[i] + beta[i - 1]

    return x


def exact(x, t):
    return np.exp(-np.pi * t) * np.sin(np.pi * x / 4)

def g(x):
    return exact(x, 0)

def mu(t):
    return exact(ax, t)

def be(t):
    return exact(bx, t)




ax, bx = 0, 4
at, bt = 0, 0.1

h = 0.01
tau = 0.01
a = 16 / np.pi

n = int((bx-ax)/h)
m = int((bt-at)/tau)

u = np.zeros((m+1, n+1))
xl = np.linspace(ax, bx, num=n+1)
tl = np.linspace(at, bt, num=m+1)

u[0] = g(xl)
u[:, 0] = mu(tl)
u[:, -1] = be(tl)



def explicit():
    var = a * tau / h ** 2
    for ti in range(m):
        for xi in range(1, n):
            u[ti+1, xi] = var * (u[ti, xi-1] - 2*u[ti, xi] + u[ti, xi+1]) + u[ti, xi]
        
    print(max(abs(u[-1] - exact(xl, bt))))
            
def implicit():
    var = a * tau / h ** 2
    for ti in range(m):
        al = np.zeros(n+1)
        bl = np.zeros(n+1)
        cl = np.zeros(n+1)
        
        al[1:-1] = var
        bl[:] = -(2 * var + 1)
        cl[1:-1] = var
        
        bl[0] = 1
        bl[-1] = 1
        
        u[ti+1] = TDMA(al, bl, cl, -u[ti])
        
    print(max(abs(u[-1] - exact(xl, bt))))

def kr_nik():
    var = a * tau / (2 * h ** 2)
    for ti in range(m):
        al = np.zeros(n+1)
        bl = np.zeros(n+1)
        cl = np.zeros(n+1)
        
        al[1:-1] = var
        bl[:] = -(2 * var + 1)
        cl[1:-1] = var
        
        bl[0] = 1
        bl[-1] = 1
        
        f = -u[ti] 
        for xi in range(1, n):
            f[xi] += -var * (u[ti, xi-1] - 2*u[ti, xi] + u[ti, xi+1])
        
        u[ti+1] = TDMA(al, bl, cl, f)
        
    print(max(abs(u[-1] - exact(xl, bt))))


explicit()
implicit()
kr_nik()

# print(max(abs(u[-1] - exact(xl, bt))))

plt.plot(xl, exact(xl, bt))
plt.plot(xl, u[-1])

plt.show()
