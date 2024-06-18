import numpy as np
import math
# import matplotlib.pyplot as plt
import sympy as sy




eps1, eps2, eps3 = sy.symbols('eps1 eps2 eps3')
a12, a13, a23 = sy.symbols('a12 a13 a23')
k12, k13, k23 = sy.symbols('k12 k13 k23')

delt = sy.symbols('delt')
m12, m13, m23 = sy.symbols('m12 m13 m23')
v12, v13, v23 = sy.symbols('v12 v13 v23')


def static_point():
    ld = k13 - k12 * k23
    if ld != 0:
        return np.array([
            (eps3 * a12 - eps1 * a23 * k23 + eps2 * k23 * a13) / (a12 * a13 * ld),
            (eps1 * a13 * k13 - eps2 * a13 * k13 - eps3 * a12 * k12) / (a12 * a23 * ld),
            (eps3 * a12 * k12 + eps2 * a13 * k13 - eps1 * k23 * k12 * k23) / (a13 * a23 * ld),
        ])
    else:
        x3 = 1
        return np.array([
            (a23 * x3 - eps2) / (a12 * k12),
            (-a13 * x3 + eps1) / a12,
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
#     [0, 0, -a23, -eps2],
#     [0, k23 * a23, 0, eps3]
# ])

# 3
# m = sy.Matrix([
#     [0, 0, -a13, -eps1],
#     [0, 1, 0, 0],
#     [k13 * a13, 0, 0, eps3]
# ])

# 4
# m = sy.Matrix([
#     [0, -a12, 0, -eps1],
#     [k12 * a12, 0, 0, -eps2],
#     [0, 0, 1, 0]
# ])

# 5
# m = sy.Matrix([
#     [0, -a12, -a13, -eps1],
#     [k12 * a12, 0, -a23, -eps2],
#     [k13 * a13, k23 * a23, 0, eps3]
# ])


# m = sy.Matrix([
#     [eps1 - a12 * x[1] - a13 * x[2], -a12 * x[0], -a13 * x[0]],
#     [k12 * a12 * x[1], eps2 + k12 * a12 * x[0] - a23 * x[2], -a23 * x[1]],
#     [k13 * a13 * x[2], k23 * a23 * x[2], -eps3 + k13 * a13 * x[0] + k23 * a23 * x[1]]
# ])

x = (x1,x2,x3)
m = sy.Matrix([
       ((-1 * x[0] + 10) * x[0] - 2 * x[1] * x[0] - 3 * x[2] * x[0]),
        ((-3 * x[1] + 9) * x[1] + (1 * x[0] - 5) * x[1] - 1 * x[2] * x[1]),
        ((1 * x[0] - 3) * x[2] + (1 * x[1] - 4) * x[2]),
])

# m = sy.Matrix([
#     ((-eps1 * x[0] + delt) - v12 * x[1] - v13 * x[2]) ,
#     ((k12 * x[0] - m12) - v23 * x[2]) ,
#     ((k13 * x[0] - m13) + (k23 * x[1] - m23) )
# ])

    
x = sy.solve(m,x1, x2, x3 )
sy.pprint(x)
print(x)
x = x[3]
print(x)

# m = sy.Matrix([
#     [(-2*eps1 * x[0] + delt) - v12 * x[1] - v13 * x[2],  - v12 * x[0], - v13 * x[0]],
#     [k12 * x[1], (k12 * x[0] - m12) - v23 * x[2], -v23 * x[1]],
#     [k13 * x[2], k23 * x[2], (k13 * x[0] - m13) + (k23 * x[1] - m23)]
# ])

# m = sy.Matrix([
#     [(-eps1 * x[0] ),  - v12 * x[0], - v13 * x[0]],
#     [0, (k12 * x[0] - m12) - v23 * x[2], 0],
#     [k13 * x[2], k23 * x[2], 0]
# ])



# m = sy.Matrix([
#     [(-eps1 * x[0] ),  - v12 * x[0], - v13 * x[0]],
#     [k12 * x[1], 0, -v23 * x[1]],
#     [k13 * x[2], k23 * x[2], 0]
# ])

m = sy.Matrix([
    [(-1 * x[0] ),  - 2 * x[0], - 3 * x[0]],
    [1 * x[1], -3 * x[1], -1 * x[1]],
    [1 * x[2], 1 * x[2], 0]
])

eig = m.eigenvals()
for v in eig:
    print(v.evalf())
    # print(v)
    print()

# lam = sy.symbols('lambda')
# p = m.charpoly(lam)
# print(p.as_expr(), '\n') 
# print(sy.factor(p.as_expr()))