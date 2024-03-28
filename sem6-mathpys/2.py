# Var 6

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation


total_k = 5

def f(x):
    return x * (1-x) ** 4

def f2(x):
    return f(x) ** 2

def y(x, k):
    return np.sin( np.pi * x * (0.5 + k) )

def y_t(k):
    def y_(x):
        return y(x, k) * f(x)
    return y_

def y_b(k):
    def y_(x):
        return y(x, k) ** 2
    return y_

def series(x, c):
    return sum(c[k] * y(x, k)  for k in range(total_k))


c = []
for k in range(total_k):
    c_t = scipy.integrate.quad(y_t(k), 0, 1)[0]
    # c_b = scipy.integrate.quad(y_b(k), 0, 1)[0]
    c_b = 0.5
    c.append(c_t / c_b)


int1 = scipy.integrate.quad(f2, 0, 1)[0]
sum1 = sum(np.array(c) ** 2)/2
print(int1, sum1, abs(int1 - sum1))


x = np.linspace(0, 1, 1000)
y = series(x, c)


plt.plot(x, y, 'c')
plt.plot(x, f(x), 'g')

plt.legend(["Fourier", "Exact"])
plt.show()

