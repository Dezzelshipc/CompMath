import numpy as np
import matplotlib.pyplot as plt
import scipy as sc

# var 25
def exact(x):
    return np.log(x**2 + x + 1)
    # return 0.934 - 0.988 * x**2 + 0.054 * x**4

def p(x):
    return 1 / (x**2 + x + 1)
    # return 1

def q(x):
    return 0
    # return (1+x**2)

def f(x):
    return (2 - 2*x**2)/(x**2 + x + 1)**2
    # return -1


def phi(x, k):
    if k == 0:
        return x
    else:
        return np.sin(2 * np.pi * k * x) / (2 * np.pi * k)
    
def phi_p(x, k):
    if k == 0:
        return 1
    else:
        return np.cos(2 * np.pi * k * x)

def u(x, c):
    return sum(c[i] * phi(x, i) for i in range(len(c)))

def part_p(i,j):
    return lambda x: p(x) * phi_p(x, j) * phi_p(x, i)

def part(i, j):
    return lambda x: -q(x) * phi(x, j) * phi(x, i)

def right(i):
    return lambda x: -f(x) * phi(x, i) + q(x) * phi(x, 0) * phi(x, i) - p(x) * phi_p(x, 0) * phi_p(x, i)

a, b = 0, 1
n = 3

matrix = np.zeros((n-1,n-1))
values = np.zeros(n-1)
for i in range(1, n):
    for j in range(1, n):
        matrix[i - 1, j - 1] = sc.integrate.quad(part_p(i,j), a, b)[0] + sc.integrate.quad(part(i,j), a, b)[0]
    
    values[i-1] = sc.integrate.quad(right(i), a, b)[0]
    
# print(matrix)
# print(values)
c = np.linalg.solve(matrix, values)
c = np.append([1], c)

print(c)
num = 1000
x = np.linspace(a, b, num+1)
plt.plot(x, exact(x))

plt.plot(x, u(x, c))

# for i in range(n):
#     plt.plot(x, c[i]*phi(x, i))


plt.show()