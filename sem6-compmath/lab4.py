import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def solve(matrix: np.matrix, values: np.array, eps: float = 1e-5, w: float = 1, iter_max=1e5):
    if w <= 0 or w >= 2:
        exit(f"Invalid value of {w = }. w must be in (0, 2)")
    x = values.copy()
    iterations = 0
    while True:
        # print(x)
        iterations += 1
        x_prev = x.copy()
        for i in range(len(matrix)):
            x[i] = (1 - w) * x[i] + w / matrix[i, i] * (values[i] - sum(matrix[i, j] * x[j] for j in range(i)) - sum(
                matrix[i, j] * x[j] for j in range(i + 1, len(matrix))))
        norm = np.linalg.norm(x - x_prev, ord=1)
        if norm < eps or iterations > iter_max:
            break
        print(iterations, norm, "\r")

    return x, iterations

def g(x, y):
    return -x*y*(x-1)*(y-1)*(2*y+x)
    # return 0

def phi(x, y):
    return 0
    # return x + y**2



ax, bx = 0, 1
ay, by = 0, 1
n = 20
m = 20

hx = bx/n
hy = by/m
xl = np.linspace(ax, bx, num=n+1)
yl = np.linspace(ay, by, num=m+1)

matrix = np.zeros(((n+1)**2, (m+1)**2))
values = np.zeros((n+1, m+1))

for i in range(1,n):
    for j in range(1,m):
        mat = np.zeros((n+1, m+1))
        
        mat[i+1, j] = mat[i-1, j] = 1/hx**2
        mat[i, j+1] = mat[i, j-1] = 1/hy**2
        mat[i, j] = -2 * (1/hx**2 + 1/hy**2)
        
        matrix[i * (n+1) + j] = mat.flatten()
        
        values[i, j] = g(xl[i], yl[j])

for i in range(n+1):
    for j in [0, m]:
        mat = np.zeros((n+1, m+1))
        
        mat[i, j] = 1
        
        matrix[i * (n+1) + j] = mat.flatten()
        
        values[i, j] = phi(xl[i], yl[j])

for i in [0, n]:
    for j in range(1,m):
        mat = np.zeros((n+1, m+1))
        
        mat[i, j] = 1
        
        matrix[i * (n+1) + j] = mat.flatten()
        
        values[i, j] = phi(xl[i], yl[j])

values = values.flatten()

# print(matrix)
# print(values)

u, it = solve(matrix, values, eps=1e-5)
print(it)
uu = u.reshape(m+1, n+1)


xx, yy = np.meshgrid(xl, yl)
        
fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
ax.plot_surface(xx, yy, uu, 
                cmap=cm.plasma,
                linewidth=0, antialiased=True)
# ax.set_title(leg[i])
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('u')

# print(cm.cmaps_listed)
plt.show()