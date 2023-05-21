import math
import numpy as np


def f(t):
    # return t**2
    # return t**2 - 1
    return (t - 1e-5) * (t + 1e-5)
    # return 1.2 * t ** 2 - np.sin(10 * t)
    # return (t - 2) * (t + 1) * (t - 5)


def df(t):
    return 2 * t
    # return 2.4 * t - 10 * math.cos(10 * t)
    # return 3 * (1 - 4 * t + t ** 2)


eps = 1e-15
a, b = -10, 10
n = 100

intervals = [(a, b)]
intervals2 = []
zeros = []

for _ in range(n):
    intervals2 = []
    xs = []
    for inter in intervals:
        intervals3 = set()
        for c in range(1, 3):
            xs = np.linspace(*inter, 10 ** c)
            fs = f(xs)
            minint = [(a,b)]
            for i in range(len(xs) - 1):
                if fs[i] * fs[i + 1] < 0:
                    intervals3.add((xs[i], xs[i + 1]))
                elif (i < len(xs) - 2) and abs(fs[i]) > abs(fs[i+1]) and abs(fs[i+1]) < abs(fs[i+2]):
                    intervals3.add((xs[i], xs[i + 1]))
                    intervals3.add((xs[i+1], xs[i + 2]))

            if len(intervals3) > 0:
                break
        intervals2.extend(intervals3)

    intervals = intervals2

for interval in intervals:
    x, x_prev = interval
    while abs(x - x_prev) >= eps:
        x_prev = x
        x = x_prev - f(x_prev) / df(x_prev)

    zeros.append(x)

print(zeros)
