import numpy as np
import matplotlib.pyplot as plt


def runge_kutta(function, y0: float, a: float, b: float, h: float):
    num = int((b - a) / h + 1)
    x_a = np.linspace(a, b, num=num, endpoint=False)
    y_a = [y0] * num

    for i in range(num - 1):
        k0 = function(x_a[i], y_a[i])
        k1 = function(x_a[i] + h / 2, y_a[i] + h * k0 / 2)
        k2 = function(x_a[i] + h / 2, y_a[i] + h * k1 / 2)
        k3 = function(x_a[i] + h, y_a[i] + h * k2)
        y_a[i + 1] = y_a[i] + h / 6 * (k0 + 2 * k1 + 2 * k2 + k3)

    return x_a, np.array(y_a)


KC = 276
sigma = 5.67e-8

T_l = 190 + KC
T_u = 200 + KC


is_turned = True
def H1(T):
    global is_turned

    if T > T_u:
        is_turned = False
    elif T < T_l:
        is_turned = True
    
    return int(is_turned)

def H0(T):
    return 1.

H = ()

leg = []
def utug(P, m, c, S, k):
    def dTdt(t, T):
        return (P * H(T) - k * S * (T - T0) - sigma * S * (T**4 - T0**4)) / (c * m)
    
    x = np.linspace(a, b, n)

    x, y = runge_kutta(dTdt, T0, a, b, (b-a)/n)
    # y -= KC
    leg.append(f"{P = }, {m = }, {c = }, {S = }, {k = }")
    plt.plot(x, y)

a, b = 0, 250
n = 10000


P = 3000
m = 0.5
c = 897  # Алюминий
S = 0.4
k = 2
T0 = 20 + KC

def roots():
    def diff(T):
        return (-4 * sigma * S * T**3 - k * S) / (c*m)

    coeff = np.zeros(5)
    coeff[0] = sigma
    coeff[3] = k
    coeff[4] = -sigma*T0**4 - k*T0 - P/S


    for root in np.roots(coeff):
        print(root, diff(root))

def without_term():
    global H
    H = H0
    utug(P, m, c, S, k)
    utug(P, 1.0, c, S, k)
    utug(P, m, 554, S, k) # Чугун
    utug(P, m, c, S/2, k)
    utug(2500, m, c, S, k)
    utug(P, m, c, S, 4)

    plt.legend(leg)
    
    plt.plot([a, b], [600, 600], 'y--')
    plt.savefig("./sem6-matmodelling/utug1.pdf")

def with_trem(P, m, c, S, k):
    global H, T_l, T_u
    H = H1
    leg2 = []
    def al2():
        leg2.append(f"T_min = {T_l}, T_max = {T_u}")

    T_l = 490
    T_u = 500
    al2()

    utug(P, m, c, S, k)
    T_l = 530
    T_u = 560
    al2()
    utug(P, m, c, S, k)

    T_l = 510
    T_u = 515
    al2()
    utug(P, m, c, S, k)
    
    plt.legend(leg2)
    plt.savefig("./sem6-matmodelling/utug2.pdf")



plt.xlabel('t - Время')
plt.ylabel('T(t) - Температура в кельвинах')

# without_term()
with_trem(2500, 1, c, S/2, 2)

plt.xlabel('t - Время')
plt.ylabel('T(t) - Температура в кельвинах')
plt.xlim([a,b])
plt.show()