import numpy as np
import matplotlib.pyplot as plt
from c1 import *


leg = []
def utug1(P, m, c, S, k, tp = 0, T_l = 190 + KC, T_u = 200 + KC):
    x, y = utug(P, m, c, S, k, tp, T_l, T_u)
    leg.append(f"{P = }, {m = }, {c = }, {S = }, {k = }")
    plt.plot(x, y)


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
    utug1(P, m, c, S, k)
    utug1(P, 1.0, c, S, k)
    utug1(P, m, 554, S, k) # Чугун
    utug1(P, m, c, S/2, k)
    utug1(2500, m, c, S, k)
    utug1(P, m, c, S, 4)

    plt.legend(leg)
    
    plt.plot([a, b], [600, 600], 'y--')
    plt.savefig("./sem6-matmodelling/utug1.pdf")

def with_trem(P, m, c, S, k):
    leg2 = []
    def al2():
        leg2.append(f"T_min = {T_l}, T_max = {T_u}")
        plt.plot([a, b], [T_l-1]*2, '--', color='#A0A0A0')
        plt.plot([a, b], [T_u+1]*2, '--', color='#A0A0A0')

    T_l = 490
    T_u = 500
    al2()

    utug1(P, m, c, S, k, 1, T_l, T_u)
    T_l = 530
    T_u = 560
    al2()
    utug1(P, m, c, S, k, 1, T_l, T_u)

    T_l = 510
    T_u = 515
    al2()
    utug1(P, m, c, S, k, 1, T_l, T_u)
    
    plt.legend(leg2)
    plt.savefig("./sem6-matmodelling/utug2.pdf")



plt.xlabel('t - Время')
plt.ylabel('T(t) - Температура в кельвинах')

# without_term()
with_trem(3000, 0.5, c, S, 2)
# with_trem(2500, 0.4, 554, S/2, 2)
# with_trem(2500, 1, c, S/2, 2)

plt.xlabel('t - Время')
plt.ylabel('T(t) - Температура в кельвинах')
plt.xlim([a,b])
plt.show()