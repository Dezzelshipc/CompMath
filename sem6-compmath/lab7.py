import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import sympy as sy

# var 25
def exact(x):
    return np.log(x**2 + x + 1)
    # return 0.934 - 0.988 * x**2 + 0.054 * x**4
    # return x + np.exp(1) / (np.exp(2) - 1) * (np.exp(-x) - np.exp(x))

def p(x):
    return 1 / (x**2 + x + 1)
    # return 1
    # return 0

def q(x):
    return 0
    # return (1+x**2)
    # return -1

def f(x):
    return (2 - 2*x**2)/(x**2 + x + 1)**2
    # return -1
    # return -x



def phi(x, k):
    return -x**(k+1) / (k+1) + x**(k+2) / (k+2) if k > 0 else x
    # return x**k * (1 - x)  if k > 0 else 0


def L(x, ff):
    x_ = sy.Symbol('x')
    u0 = ff(x_)
    u1 = sy.diff(u0, x_)
    u2 = sy.diff(u1, x_)
    ret = u2 + p(x_) * u1 + q(x_) * u0
    return ret.subs(x_, x)
    


def u(x, c):
    return sum(c[i] * phi(x, i) for i in range(len(c)))


a, b = 0, 1
n = 3

matrix = np.zeros((n-1,n-1))
values = np.zeros(n-1)

for i in range(1, n):
    for j in range(1, n):
        integrant = lambda x: phi(x, i) * L(x, lambda xx: phi(xx, j))
        int1 = sc.integrate.quad(integrant, a, b)
        
        matrix[i - 1, j - 1] = int1[0] 
        
    
    integrant = lambda x: phi(x, i) *(f(x) - L(x, lambda xx: phi(xx, 0)))
    intv = sc.integrate.quad(integrant, a, b)
    values[i-1] = intv[0]

    
# print(matrix)
# print(values)
c = np.linalg.solve(matrix, values)
c = np.append([1], c)

# print(c)
num = 1000
x = np.linspace(a, b, num+1)

uu = u(x, c)
plt.plot(x, uu, 'r')

plt.plot(x, exact(x), 'b--')

plt.legend(["Решение", "Точное"])

plt.figure(123)
udiff = abs( exact(x) - uu )

plt.plot(x, udiff)


plt.show()