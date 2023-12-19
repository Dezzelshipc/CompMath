import numpy as np
import lab3

# A = np.matrix([[0., 1., -1.],
#                [-1., 0., 1.],
#                [1., -1., 0]])

A = np.matrix([[1., 5., 9., 3., 2., 5.],
               [2., 9., 3., 5., 3., 6.],
               [7., 4., 6., 10., 2., 5.],
               [5., 9., 3., 3., 4., 6.],
               [6., 3., 8., 9., 6., 7.],
               [4., 6., 2., 3., 4., 7.],
               [9., 1., 2., 3., 5., 9.],
               [2., 3., 5., 2., 6., 10.]]).T

min_a = A.min()
if min_a < 0:
    A += abs(min_a) + 1

print(A)
maxmin = A.min(axis=1).max()
minmax = A.max(axis=0).min()
print(f'{maxmin = } {minmax = }')

n = len(A)  # 6
m = len(A.T)  # 8

b = np.ones((1, n))
f = np.ones((1, m))
# print(b, f)


pv, x = lab3.simplex_dual(A, b, f)
qv, y = lab3.simplex(A, b, f)
print(x, y)
v = 1 / pv
v2 = 1 / qv
print(v, v2)
if abs(v-v2) > 1e-10:
    exit("Simplex method error")

print(f'p = {x * v = }\n{sum(x * v)=}\nq = {y * v = }\n{sum(y * v)=}')
