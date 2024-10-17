import numpy as np
import matplotlib.pyplot as plt


def exact(x):
    # return np.sin(x)
    return np.exp(x)

a = 0
b = 7
eps = 1e-3
n = 100
h = (b - a)/n

x = np.linspace(a, b, n+1)

# f = lambda x: x
f = lambda x: 1
# K = lambda x, s: -(x - s)
K = lambda x, s: 1
phi = np.zeros(n+1)
for i in range(n+1):
    phi[i] = f(x[i])

Phi = []
Phi.append(phi)
k = 0
flag = 1
y_k1 = np.zeros(n+1)

while flag > eps:
    phi = np.zeros(n+1)
    Phi.append(phi)
    for i in range(n+1):
        for j in range(1, i):
            Phi[k+1][i] += h * (K(x[i], x[j]) * Phi[k][j])
        if i == 0:
            Phi[k+1][i] = 0
        else:
            Phi[k+1][i] += h / 2 * (K(x[i], x[0]) * Phi[k][0] + K(x[i], x[i]) * Phi[k][i])

    y_k = y_k1.copy()
    y_k1 = np.zeros(n+1)
    for i in range(k):
        y_k1 += np.array(Phi[i])


    if k > 1:
        flag = np.linalg.norm(y_k - y_k1, np.inf) / np.linalg.norm(y_k, np.inf)
        print(flag)

    k += 1



print('Количество итераций:', k)

e = np.zeros(n+1)
for i in range(n+1):
    e[i] = exact(x[i])

error = np.zeros(n+1)
error = abs(e - y_k1)

print(y_k1[3],y_k1[-1])

plt.figure(1)

plt.plot(x,e)
plt.plot(x,y_k1)


plt.legend(['Точное решение', 'Метод простой итерации, ур.Вольтерры 2-го рода.'])

plt.figure(2)
plt.plot(x,error)
plt.legend(['Ошибка'])

plt.show()