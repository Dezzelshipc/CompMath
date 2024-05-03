import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

# var 13
def g(x, t):
    return t* x**2*(1-x)
    # return 2*t*x**2*(1-x)
    # return -x**2*(1-x)
    # return 0

def phi(x):
    # return x*(1-x)
    return 2*x*(1-x)
    # return x*(x**2-1)
    # return 0

def psi(x):
    return x**3 - x**2
    # return -x**2 + x**3
    # return 0

def ga0(t):
    # return 0
    return 3*t

def ga1(t):
    # return 0
    return t*(1-t)


a = 2
ax, bx = 0, 1
at, bt = 0, 1

n = 200
m = 1000

h = (bx - ax)/n
tau = (bt - at)/m

u = np.zeros((m+1, n+1))
xl = np.linspace(ax, bx, num=n+1)
tl = np.linspace(at, bt, num=m+1)

u[0] = phi(xl)
u[:,0] = ga0(tl)
u[:,-1] = ga1(tl)
# u[1] = u[0] + tau * psi(xl)

for xi in range(1,n):
    u[1,xi] = u[0,xi] + tau**2 / 2 * (a**2 / h ** 2 * (u[0, xi-1] - 2*u[0,xi] + u[0,xi+1]) + g(xl[xi], tl[0])) + tau * psi(xl[0])

for ti in range(1,m):
    for xi in range(1, n):
        u[ti+1,xi] = a ** 2 * tau ** 2 / h ** 2 * (u[ti, xi-1] - 2*u[ti, xi] + u[ti, xi+1]) + tau**2 * g(xl[xi], tl[ti]) + 2*u[ti,xi] - u[ti-1,xi]

  

# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# xx, tt = np.meshgrid(xl, tl)
        
# ax.plot_surface(xx, tt, u, 
#                 cmap=cm.plasma,
#                 linewidth=0, antialiased=True)

# ax.set_xlabel('x')
# ax.set_ylabel('t')
# ax.set_zlabel('u')

# plt.imshow(u, cmap=cm.plasma, origin='lower', extent=[ax,bx,at,bt])


from matplotlib.animation import FuncAnimation

fig, ax = plt.subplots()
plot, = plt.plot(xl, u[0])

ax.set_ylim([np.min(u)-2*h, np.max(u)+2*h])

def init():
    return plot,

def update(ti):
    plot.set_data(xl, u[ti])
    return plot,

frames, interval = range(m+1), 1

ani = FuncAnimation(fig, update, frames=frames,
                    init_func=init, blit=True, interval=interval)

plt.show()