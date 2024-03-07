import sympy as sp
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy as sci

np.set_printoptions(1)

# Алюминий
P = 2000.
m = 0.5
c = 920.
s = 0.4
k = 10.
T0 = 300.
sigma = 1.380649e-23

lastT = T0
def H(T, limTop, limBottom):
    global lastT
    limTop += 276
    limBottom += 276

    res = 0.
    if lastT > T:
        if T < limBottom:
            res = 1.
    else:
        if T > limTop:
            res = 0.
        else:
            res = 1.

    lastT = T
    return res
    # return math.tanh((lim + 276)**100 / T**100)

def dTdt(t, T):
    return (P * H(T, 350, 270) - k * s * (T - T0) - sigma * s * (T**4 - T0**4)) / (c * m)

a, b = 0, 200

t = np.linspace(a, b, 1000)
sol = sci.integrate.solve_ivp(dTdt, [a, b], [T0], t_eval=t)


print(t)
print(sol.y[0])
print(np.vectorize(dTdt)(sol.y[0], t))

plt.plot(t, sol.y[0] - 276)
plt.show()
