# Var 6

import numpy as np
import scipy
import matplotlib.pyplot as plt


total_k = 5
a, b = 0, 1
# a, b = 0, 2
# a, b = 0, 0.8
# a, b = 0, 0.6



def f(x):
    return  x * (1-x) ** 4
    # return  (x ** 2) * (2-x)
    # return x** 2 * np.log(1.8 - x)
    # return np.cos(x*(0.6-x))

def f2(x):
    return f(x) ** 2

def y(x, k):
    f = lambda x:  np.sin(np.pi * x * (0.5 + k) )
    # f = lambda x: np.cos( np.pi * x * (0.5 + k) / 2 )
    # f = lambda x: np.cos(5/4 * np.pi* (1/2 + k) * x)
    # f = lambda x: np.cos(5*np.pi * k *x / 3)

    # n = scipy.integrate.quad(f, a, b)[0]
    n = 1/2
    # n = 0.3 if k != 0 else 0.6
    return f(x) / np.sqrt(n)
    
    

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
    c_t = scipy.integrate.quad(y_t(k), a, b)[0]
    # c_b = scipy.integrate.quad(y_b(k), a, b)[0]
    c_b = 1
    c.append(c_t / c_b)


print(c)
int1 = scipy.integrate.quad(f2, a, b)[0]
sum1 = np.sum(np.array(c) ** 2)
print(int1, sum1, abs(int1 - sum1))


x = np.linspace(a, b, 1000)
y = series(x, c)


plt.plot(x, y, 'c')
plt.plot(x, f(x), 'g')

plt.legend(["Fourier", "Exact"])

# plt.savefig("fourier.pdf", format="pdf")
plt.show()

