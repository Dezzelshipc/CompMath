import numpy as np
import math
import matplotlib.pyplot as plt
import sympy as sy




ksi1, ksi2, ksi3 = sy.symbols('ksi1 ksi2 ksi3')
a12, a13, a23 = sy.symbols('a12 a13 a23')
k12, k13, k23 = sy.symbols('k12 k13 k23')


def static_point():
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


x1, x2, x3 = sy.symbols('x1 x2 x3')

# 1
# m = sy.Matrix([
#     [1, 0, 0, 0],
#     [0, 1, 0, 0],
#     [0, 0, 1, 0]
# ])

# 2
# m = sy.Matrix([
#     [1, 0, 0, 0],
#     [0, 0, -a23, -ksi2],
#     [0, k23 * a23, 0, ksi3]
# ])

# 3
# m = sy.Matrix([
#     [0, 0, -a13, -ksi1],
#     [0, 1, 0, 0],
#     [k13 * a13, 0, 0, ksi3]
# ])

# 4
# m = sy.Matrix([
#     [0, -a12, 0, -ksi1],
#     [k12 * a12, 0, 0, -ksi2],
#     [0, 0, 1, 0]
# ])

# 5
m = sy.Matrix([
    [0, -a12, -a13, -ksi1],
    [k12 * a12, 0, -a23, -ksi2],
    [k13 * a13, k23 * a23, 0, ksi3]
])


x = (sy.linsolve(m,(x1, x2, x3 ))).args[0]
print(x, '\n')


m = sy.Matrix([
    [ksi1 - a12 * x[1] - a13 * x[2], -a12 * x[0], -a13 * x[0]],
    [k12 * a12 * x[1], ksi2 + k12 * a12 * x[0] - a23 * x[2], -a23 * x[1]],
    [k13 * a13 * x[2], k23 * a23 * x[2], -ksi3 + k13 * a13 * x[0] + k23 * a23 * x[1]]
])

eig = m.eigenvals()
for k,v in eig.items():
    print(k)
    print()
    print(v)
    print()
    
lam = sy.symbols('lambda')
p = m.charpoly(lam)
print(p.as_expr(), '\n') 
print(sy.factor(p.as_expr()))