import numpy as np
import matplotlib.pyplot as plt


def exact(x):
    return np.exp(x)

a = 0
b = 7
eps = 1e-5
n = 1000

x, h = np.linspace(a, b, n+1, retstep=True)

f = lambda x: 1
K = lambda x_i, x_j: 1

Phi = [np.zeros(n+1) + f(x)]
k = 0
flag = 1
y_k = np.zeros(n+1) + Phi[0]

while flag > eps:
    Phi.append(np.zeros(n+1))
    for i in range(n+1):
        for j in range(1, i):
            Phi[-1][i] += h * (K(x[i], x[j]) * Phi[k][j])
        if i == 0:
            Phi[-1][i] = 0
        else:
            Phi[-1][i] += h / 2 * (K(x[i], x[0]) * Phi[k][0] + K(x[i], x[i]) * Phi[k][i])


    y_k += np.array(Phi[-1])

    if k > 1:
        flag = max(abs(Phi[-1])) / max(abs(y_k))
        print(flag)

    k += 1


print('Количество итераций:', k)

e = exact(x)

error = abs(e - y_k)

plt.figure(1)

y = np.zeros_like(y_k)


for i in range(len(Phi)):
    y += Phi[i]
plt.plot(x,y)

plt.plot(x,e,"--")
# plt.ylim((-2,2))

plt.legend(['Точное решение', 'Метод простой итерации, ур.Вольтерры 2-го рода.'])

plt.figure(2)
plt.plot(x,error)
plt.legend(['Ошибка'])

plt.show()