import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


# var 13
def g(x, t):
    return t* x**2*(1-x)
    # return 2*t*x**2*(1-x)

def phi(x):
    return x*(1-x)
    # return 2*x*(1-x)

def psi(x):
    return x**3 - x**2
    # return -x**2 + x**3

def ga0(t):
    return 0

def ga1(t):
    return 0


a = 1

ax, bx = 0, 1
at, bt = 0, 1

n = 100
m = 100

h = (bx - ax)/n
tau = (bt - at)/m

u = np.zeros((m+1, n+1))
xl = np.linspace(ax, bx, num=n+1)
tl = np.linspace(at, bt, num=m+1)

u[0] = phi(xl)
u[:,0] = ga0(tl)
u[:,-1] = ga1(tl)
u[1] = u[0] + tau * psi(xl)

for ti in range(2,m):
    for xi in range(1, n):
        u[ti+1,xi] = a ** 2 * tau ** 2 / h ** 2 * (u[ti, xi-1] - 2*u[ti, xi] + u[ti, xi+1]) + tau**2 * g(xl[xi], tl[ti]) + 2*u[ti,xi] - u[ti-1,xi]
        
xx, tt = np.meshgrid(xl, tl)
        
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xx, tt, u, 
                cmap=cm.plasma,
                linewidth=0, antialiased=True)

ax.set_xlabel('x')
ax.set_ylabel('t')
ax.set_zlabel('u')


plt.show()