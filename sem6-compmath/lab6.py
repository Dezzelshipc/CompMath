# Ritz method
import numpy as np
import matplotlib.pyplot as plt
import scipy as sc
import sympy as sy

# var 25
def exact(x):
    return np.log(x**2 + x + 1)

def P(x):
    return 1 / (x**2 + x + 1)

def Q(x):
    return 0

def F(x):
    return (2 - 2*x**2)/(x**2 + x + 1)**2

def p(x):
    integ = sc.integrate.quad(P, 0, x)[0]
    return np.exp(integ)

def q(x):
    return p(x) * Q(x)

def f(x):
    return p(x) * F(x)


def phi(x, k):
    return x**k * (1 - x)  if k > 0 else (1- np.log(3)) * x**2 + (2*np.log(3) - 1) * x
    # return x**k * (1 - x)  if k > 0 else exact(1) * x
    # return x**(k+1) / (k+1) - x**(k+2) / (k+2) if k > 0 else (1- np.log(3)) * x**2 + (2*np.log(3) - 1) * x
    
    
def phi_p(x, k):
    x_ = sy.Symbol('x')
    p = phi(x_, k)
    p = sy.diff(p, x_)
    return p.subs(x_, x)
    


def u(x, c):
    return sum(c[i] * phi(x, i) for i in range(len(c)))

def part_p(i,j):
    return lambda x: p(x) * phi_p(x, j) * phi_p(x, i)

def part(i, j):
    return lambda x: -q(x) * phi(x, j) * phi(x, i)

def right(i):
    return lambda x: -f(x) * phi(x, i) + q(x) * phi(x, 0) * phi(x, i) - p(x) * phi_p(x, 0) * phi_p(x, i)

a, b = 0, 1
n = 10

matrix = np.zeros((n-1,n-1))
values = np.zeros(n-1)

x_ = sy.Symbol('x')

for i in range(1, n):
    for j in range(1, n):
        int1 = sc.integrate.quad(part_p(i,j), a, b)
        int2 = sc.integrate.quad(part(i,j), a, b)
        
        matrix[i - 1, j - 1] = int1[0] + int2[0]
        
    
    intv = sc.integrate.quad(right(i), a, b)
    
    values[i-1] = intv[0]
    
    
# print(matrix)
# print(values)
c = np.linalg.solve(matrix, values)
c = np.append([1], c)

print(c)
num = 1000
x = np.linspace(a, b, num+1)

uu = u(x, c)
plt.plot(x, uu)

plt.plot(x, exact(x), '--')
plt.legend(["Решение", "Точное"])

plt.figure(123)
udiff = abs( exact(x) - uu )

plt.plot(x, udiff)


plt.show()