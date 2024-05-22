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


ksi1, ksi2, ksi3 = 10, 8, 6
a12, a13, a23 = 6, 2, 0.5
k12, k13, k23 = 4, 1, 0.5

def static_point(num: int, num2 = None):
    match (num, num2):
        case (0, None):
            return [0,0,0]
        case (1, None) | (1, 2) | (2, 1):
            return [0, ksi3 / (k23 * a23), ksi2/a23]
        case (2, None) | (0, 2) | (2, 0):
            return [ksi3 / (k13 * a13), 0, ksi1/a13]
        case (3, None) | (0, 1) | (1, 0):
            return [-ksi2 / (k12 * a12), ksi1/a12, 0]
        case (4, None):
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
        ( ksi2 + k12 * a12 * x[0] - a23 * x[2] ) * x[1],
        (-ksi3 + k13 * a13 * x[0] + k23 * a23 * x[1] ) * x[2]
    ])


def plot3(tl, xl):
    plt.figure(0)
    plt.plot(tl, xl[0])
    plt.plot(tl, xl[1])
    plt.plot(tl, xl[2])

    plt.legend(["x1", "x2", "x3"])
    
def plotp(tl, xl, n1, n2):
    plt.figure(f"{n1}{n2}")
    plt.plot(xl[n1], xl[n2], 'o-', markevery=[0])
    plt.xlabel(f"x{n1+1}")
    plt.ylabel(f"x{n2+1}")
    st_point = static_point(n1, n2)
    plt.plot(st_point[n1], st_point[n2], 'o')
    
    
def plotp3(tl, xl):
    ax = plt.figure("123").add_subplot(projection='3d')
    
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')

    ax.plot(*static_point(0), 'o')
    ax.plot(*static_point(1), 'o')
    ax.plot(*static_point(2), 'o')
    ax.plot(*static_point(3), 'o')
    # ax.plot(*static_point(4), 'o')

    ax.legend(["x(0)", "x(1)", "x(2)", "x(3)", "x(4)"])

    
    ax.plot(xl[0], xl[1], xl[2], 'o-', markevery=[0])

def plotvec(n1, n2, lin1, lin2):
    plt.figure(f"vec{n1}{n2}")
    plt.xlabel(f"x{n1+1}")
    plt.ylabel(f"x{n2+1}")
    plt.plot(*static_point(n1, n2), 'o')


    grid = np.meshgrid(lin1, lin2)
    if 0 not in (n1, n2):
        vec_grid = right(0, [0, grid[0], grid[1]])
    elif 1 not in (n1, n2):
        vec_grid = right(0, [grid[0], 0, grid[1]])
    elif 2 not in (n1, n2):
        vec_grid = right(0, [grid[0], grid[1], 0])

    print([len(v) for v in grid])
    print([len(v) for v in vec_grid])

    plt.quiver(grid[0], grid[1], vec_grid[n1], vec_grid[n2], length=0.01, color = 'black')


def plotvec3(linx, liny, linz):
    ax = plt.figure("vector field").add_subplot(projection='3d')
    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('x3')

    grid = np.meshgrid(linx, liny, linz)
    vec_grid = right(0, grid)

    ax.quiver(*grid, *vec_grid, length=0.01, color = 'black')

    # ax.plot(*static_point(0), 'o')
    # ax.plot(*static_point(1), 'o')
    ax.plot(*static_point(2), 'o')
    # ax.plot(*static_point(3), 'o')

    # ax.set_xlim(-0.1, 12)
    # ax.set_ylim(-0.1, 20)
    # ax.set_zlim(-3, 3)



a, b = 0, 10
n = 10000
h = (b-a)/n

x0 = np.array([10, 10, 1])
# x0 = static_point()
tl, xl = runge_kutta(right, x0, a, b, h)
xl = xl.T

print(x0)

# plot3(tl, xl)

# plotp(tl, xl, 0, 1)
# plotp(tl, xl, 0, 2)
# plotp(tl, xl, 1, 2)

# plotp3(tl, xl)


# plotvec(1, 2, np.linspace(0.01, 50, 21), np.linspace(0.01, 50, 21))
plotvec3(np.linspace(0.01, 20, 21), 0, np.linspace(0.01, 20, 21))

# print(x0:=static_point(4))
# a2 = k12 * a12**2 * x0[0] * x0[1] + k13 * a13**2 * x0[0]*x0[2] + k23 * a23 ** 2 * x0[1] * x0[2]
# print(f"{a2=}")
# a3 = x0[0] * x0[1] * x0[2] * a12 * a13* a23 * (k13- k12 * k23)
# print(f"{a3=}")


plt.show()