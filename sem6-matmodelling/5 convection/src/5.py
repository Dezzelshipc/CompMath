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


def u0(x):
    # return np.exp(- (x - x0)**2 / d**2)
    return np.where(1 < x, np.where(x < x0, 1, 0), 0)
    
    x_m = (1+x0)/2
    kx = 1/(x_m - 1)
    
    return np.where(abs(x - x_m) < x_m - 1, 
                     np.where(x < x_m,
                        kx * x - kx,
                        -kx * x + kx + 2),
                0)

def ux0(t):
    return 0

def uxl(t):
    return 0

c = 1
l = 100

d = 1
x0 = 10*d

dx = 1
dt = 0.01

n = int(l // dx)
ax, bx = 0, n * dx

at, bt = 0, (bx - 2*x0)/c
xl = np.linspace(ax, bx, n+1)

m = int(bt // dt)
bt = m * dt
tl = np.linspace(at, bt, m+1)

u = np.zeros((m+1, n+1))

u[0] = u0(xl)
u[:,0] = ux0(tl)
u[:,-1] = uxl(tl)

def explicit(uu):
    u_ = uu.copy()
    k = c * dt / (2 * dx)
    for ti in range(m):
        for xi in range(1, n):
            u_[ti+1, xi] = u_[ti, xi] + k * (u_[ti, xi-1] - u_[ti, xi+1])
    
    return u_


def explicit2(uu):
    u_ = uu.copy()
    k = c * dt / dx
    for ti in range(m):
        for xi in range(1, n):
            u_[ti+1, xi] = u_[ti-1, xi] + k * (u_[ti, xi-1] - u_[ti, xi+1])
    
    return u_


def implicit(uu):
    u_ = uu.copy()
    k = c * dt / (2 * dx)
    for ti in range(m):
        al = np.zeros(n+1)
        bl = np.zeros(n+1)
        cl = np.zeros(n+1)
        
        al[1:-1] = -k
        bl[:] = 1
        cl[1:-1] = k
        
        bl[0] = 1
        bl[-1] = 1
        
        u_[ti+1] = TDMA(al, bl, cl, u_[ti])
        
    return u_
        
def upstream(uu):
    u_ = uu.copy()
    k = c * dt / dx
    for ti in range(m):
        for xi in range(1, n):
            u_[ti+1, xi] = u_[ti, xi] + k * (u_[ti, xi-1] - u_[ti, xi])
    
    return u_
    
u_all = [explicit(u), implicit(u), upstream(u)]
# u_all.append(explicit2(u))

u = u_all[2] # change here

s_all = [ np.sum(u_c, axis=1) for u_c in u_all ]
s_all = [ (s - s[0])/s[0] for s in s_all ]

plt.figure("Area in time")
for si in s_all:
    plt.plot(tl, si)

plt.legend(["Явная схема", "Неявная схема", "Вверх по потоку"])


show = 1

if show == 0:
    plt.figure("Time stamps")
    plt.plot(xl, u[0])
    plt.plot(xl, u[(m+1)//4])
    plt.plot(xl, u[(m+1)//2])
    plt.plot(xl, u[(m+1)//4*3])
    plt.plot(xl, u[-1])

elif show == 1:
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    plot, = plt.plot(xl, u[0])

    umin = np.min(u)
    umax = np.max(u)
    du = abs(umax - umin) / 50
    
    ax.set_ylim([umin-du, umax+du])
    # ax.set_xlabel(f"Площадь {sum(u[0])}")
    ax.set_title("0")


    def update(ti):
        plot.set_data(xl, u[ti])
        # ax.set_xlabel(f"Площадь {sum(u[ti])}")
        ax.set_title(f"Момент времени: {np.round(tl[ti], 3)}")
        
    time = 20
    frames = list(map(int, np.linspace(0, m, num = time*10)))
    
    ani = FuncAnimation(fig, update, frames=frames, interval=10)

plt.show()

