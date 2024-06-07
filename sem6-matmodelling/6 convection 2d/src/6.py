import numpy as np
from numpy import pi as PI
import matplotlib.pyplot as plt

raise NotImplementedError("Разностный метод не работает!")

def c0(x, y):
    return np.arctan( (y - 0.5) / 0.1 )


def u(x, y):
    return -PI * np.sin(2 * PI * x) * np.cos(PI * y)

def v(x,y):
    return 2 * PI * np.cos(2 * PI * x) * np.sin(PI * y)


nt = 2000
nx = 10
ny = 10

tl, dt = np.linspace(0, 0.1, nt+1, retstep=True)
xl, dx = np.linspace(0, 1, nx+1, retstep=True)
yl, dy = np.linspace(0, 1, ny+1, retstep=True)
print(dt,dx,dy)

c = np.zeros((nt+1,ny+1,nx+1))

xg2, yg2 = np.meshgrid(xl, yl)

c[0] = c0(xg2, yg2)

for t in range(0, nt):
    for y in range(0, ny+1):
        for x in range(0, nx+1):
            if (x,y) == (nx, 0):
                c[t+1,y,x] = dt * (u(xl[x], yl[y])/dx*(c[t,y,x-1] - c[t,y,x]) + v(xl[x], yl[y])/dy*(c[t,y,x] - c[t,y+1,x])) + c[t,y,x]

            elif (x,y) == (0, ny):
                c[t+1,y,x] = dt * (u(xl[x], yl[y])/dx*(c[t,y,x] - c[t,y,x+1]) + v(xl[x], yl[y])/dy*(c[t,y-1,x] - c[t,y,x])) + c[t,y,x]

            elif x == 0 or y == 0:
                c[t+1,y,x] = dt * (u(xl[x], yl[y])/dx*(c[t,y,x] - c[t,y,x+1]) + v(xl[x], yl[y])/dy*(c[t,y,x] - c[t,y+1,x])) + c[t,y,x]

            # elif x == nx or y == ny:
            else:
                c[t+1,y,x] = dt * (u(xl[x], yl[y])/dx*(c[t,y,x-1] - c[t,y,x]) + v(xl[x], yl[y])/dy*(c[t,y-1,x] - c[t,y,x])) + c[t,y,x]

            # else:
            #     c[t+1,y,x] = -dt * (u(xl[x], yl[y]) *(c[t,y,x-1] - 2 * c[t,y,x] + c[t,y,x+1])/dx**2 + v(xl[x], yl[y]) *(c[t,y-1,x] - 2*c[t,y,x] + c[t,y+1,x])/dy**2) + c[t,y,x]


# plt.quiver(xg2, yg2, u(xg2, yg2), v(xg2, yg2))


show = 1
if show == 1:
    from matplotlib.animation import FuncAnimation

    fig, ax = plt.subplots()
    plt.imshow(c[0])

    ax.set_title("0")

    def update(ti):
        plt.imshow(c[ti])
        ax.set_title(f"Момент времени: {np.round(tl[ti], 3)}")
        
    # time = 20
    # frames = list(map(int, np.linspace(0, m, num = time*10)))
    
    ani = FuncAnimation(fig, update, frames=range(0, nt, 100), interval=10)

else:
    plt.pcolormesh(xg2, yg2, c[1])

# print(c[-1])
plt.show()
