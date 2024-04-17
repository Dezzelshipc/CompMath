import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib import cm

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
a = 16 / np.pi

n = 100 # spase steps
m = 10000 # time steps

u = np.zeros((m+1, n+1))
xl = np.linspace(ax, bx, num=n+1)
tl = np.linspace(at, bt, num=m+1)

h = xl[1]-xl[0]
tau = tl[1]-tl[0]

print(tau, h**2 / (2*a**2), tau < h**2 / (2 * a**2))

u[0] = g(xl)
u[:, 0] = mu(tl)
u[:, -1] = be(tl)

ul = [[]] * 3

def explicit():
    var = a * tau / h ** 2
    for ti in range(m):
        for xi in range(1, n):
            u[ti+1, xi] = var * (u[ti, xi-1] - 2*u[ti, xi] + u[ti, xi+1]) + u[ti, xi]
    
    ul[0] = u.copy()
            
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
        
    ul[1] = u.copy()

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
      
    ul[2] = u.copy()  


explicit()
implicit()
kr_nik()

# print(max(abs(u[-1] - exact(xl, bt))))
il = range( len(tl))

for ui in ul:
    dl = []
    if len(ui):
        for i in il:
            dl.append(max(abs(ui[i] - exact(xl, tl[i]))))
    else:
        dl = [0]*len(il)
        
    plt.plot(tl, dl, 'o-', markersize=2)
    
leg = ["1. Explicit", "2. Implicit", "3. Kr-Nick"]
plt.legend(leg)
plt.xlabel("t")
plt.ylabel("max error")
plt.title(f"space steps: {n}, time steps: {m}")

xx, tt = np.meshgrid(xl, tl)
for i in range(3):
    if len(ul[i]):
        fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
        ax.plot_surface(xx, tt, ul[i], cmap=cm.plasma,
                        linewidth=0, antialiased=False)
        ax.set_title(leg[i])
        ax.set_xlabel('x')
        ax.set_ylabel('t')
        ax.set_zlabel('u')

# print(cm.cmaps_listed)
plt.show()
