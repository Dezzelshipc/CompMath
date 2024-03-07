import numpy as np
import lab3

# A = np.matrix([[0., 1., -1.],
#                [-1., 0., 1.],
#                [1., -1., 0]])

# A = np.matrix([[1., 5., 9., 3., 2., 5.],
#                [2., 9., 3., 5., 3., 6.],
#                [7., 4., 6., 10., 2., 5.],
#                [5., 9., 3., 3., 4., 6.],
#                [6., 3., 8., 9., 6., 7.],
#                [4., 6., 2., 3., 4., 7.],
#                [9., 1., 2., 3., 5., 9.],
#                [2., 3., 5., 2., 6., 10.]]).T

# A = np.matrix([[7., 8., 11., 2., 5., 9., 10., 7.],
#                [5., 3., 10., 10., 9., 10., 10., 2.],
#                [1., 3., 11., 5., 6., 4., 3., 3.],
#                [4., 6., 4., 4., 3., 10., 10., 3.],
#                [5., 8., 9., 4., 10., 2., 10., 8.],
#                [1., 9., 11., 2., 8., 1., 3., 11.]])

np.random.seed(151)
A = np.matrix(np.random.randint(1, 10, (6, 8))).astype(float)

min_a = A.min()
if min_a < 0:
    A += abs(min_a) + 1

print(A)
maxmin = A.min(axis=1).max()
minmax = A.max(axis=0).min()
print(f'{maxmin = } {minmax = }')

n = len(A)  # 6
m = len(A.T)  # 8

b = np.matrix(np.ones((1, n)))
f = np.matrix(np.ones((1, m)))
# print(b, f)


p1v, x = lab3.simplex_dual(A, b, f)
q1v, y = lab3.simplex(A, b, f)
# print(x, y)
# print(pv, np.sum(x), qv, np.sum(y))
pv = 1 / p1v
qv = 1 / q1v
print(f"{pv = }, {qv = }")
if abs(pv - qv) > 1e-10:
    exit("Simplex method error")
p = x * pv
q = y * pv
print(f'{p = }\n{sum(p)=}\n{q = }\n{sum(q)=}')

print(f"{p.T.dot(A).dot(q) = }")
