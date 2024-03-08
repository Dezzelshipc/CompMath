# Var 6

import numpy as np
import scipy
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

a = 3

def phi(x):
    return np.cos(1.5 * x) if -1 <= x <= 1 else 0 

def psi(x):
    func = 0 # 0.1 * x**2
    return func if -1 <= x <= 1 else 0 

def u(x_, t):
    return [0.5 * (phi(x - a*t) + phi(x + a*t)) + scipy.integrate.quad(psi, x - a*t, x + a*t)[0] for x in x_]




x = np.linspace(-2, 2, num=1000)

fig, ax = plt.subplots()

plot, = plt.plot(x, u(x, 0))

def init():
    return plot,

def update(frame):
    plot.set_data(x, u(x, frame))
    return plot,

# frames, interval = np.linspace(0, 2/a, 100), 40
frames, interval = [ 1/(4*a), 1/(2*a), 3/(4*a), 1/a, 2/(a) ], 1000

ani = FuncAnimation(fig, update, frames=frames,
                    init_func=init, blit=True, interval=interval)

plt.show()
