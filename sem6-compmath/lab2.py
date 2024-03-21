import numpy as np
import math
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def TDMAsolver(a, b, c, d):
    nf = len(d)  # number of equations
    ac, bc, cc, dc = map(np.array, (a, b, c, d))  # copy arrays
    for it in range(1, nf):
        mc = ac[it - 1] / bc[it - 1]
        bc[it] = bc[it] - mc * cc[it - 1]
        dc[it] = dc[it] - mc * dc[it - 1]

    xc = bc
    xc[-1] = dc[-1] / bc[-1]

    for il in range(nf - 2, -1, -1):
        xc[il] = (dc[il] - cc[il] * xc[il + 1]) / bc[il]

    return xc



# Var 5
al0, be0, ga0 = 1, -2, 1
al1, be1, ga1 = 1, 0, 0.5

# Var 1
# al0, be0, ga0 = 1, 0, 1
# al1, be1, ga1 = 1, 0, 0.5

# Test
# al0, be0, ga0 = -1, 1, 0.6
# al1, be1, ga1 = 1, 1, 4 * np.exp(3) + np.exp(4)

def exact(x):
    return 1 / (x ** 2 + 1)
    # return 1/ (x+1)
    # return np.exp(-x) + np.exp(3*x) + 0.2 *np.exp(4*x)

def p(x):
    return -(x ** 2 + 1)
    # return -x+1
    # return -2

def q(x):
    return -2*x
    # return -1
    # return -3

def f(x):
    return 2 * (3 * x**2 - 1) / (x ** 2 + 1) ** 3
    # return 2/(x+1)**3
    # return np.exp(4*x)

def ac(h, p_):
    r = p_ * h / 2
    mid = abs(r) + np.exp(-abs(r))
    return (mid - r) / h ** 2, (mid + r) / h ** 2


def solver(l, r, n):
    al, bl, cl, fl = [], [], [], []
    xl = []
    
    h = (r - l)/n

    x = l + h
    i_c = 1


    a0 = al0 - 3 * be0 / (2 * h)
    b0 = 2 * be0 / h
    c0 = -be0 / (2 * h)
    if be0 == 0:
        xl.append(l)
        bl.append(al0)
        cl.append(0)
        fl.append(ga0)
        i_c = 1
    else:
        a1, c1 = ac(h, p(x))
        
        b1 = q(x) - a1 - c1
        k = a1/a0
        
        xl.append(x)
        bl.append(b1 - k * b0)
        cl.append(c1 - k * c0)
        fl.append(f(x) - k * ga0)
        i_c = 2


    end_i = n-1 if be0 != 0 else n

    for i in range(i_c, end_i):
        x = l + h * i
        xl.append(x)
        
        ai, ci = ac(h, p(x))
        bi = q(x) - ai - ci
        
        al.append(ai)
        bl.append(bi)
        cl.append(ci)
        fl.append(f(x))


    an = be1/(2*h)
    bn = -2*be1/h
    cn = al1 + 3*be1/(2*h)
    if be1 == 0:
        xl.append(r)
        al.append(0)
        bl.append(al1)
        fl.append(ga1)
    else:
        x = (n-1) * h
        
        an1, cn1 = ac(h, p(x))
        bn1 = q(x) - an1 - cn1
        
        k = cn1 / cn
        
        xl.append(x)
        al.append(an1 - k * an)
        bl.append(bn1 - k * bn)
        fl.append(f(x) - k * ga1)
    
    yl = TDMAsolver(al, bl, cl, fl)
    
    # A = np.diag(al, -1) + np.diag(bl) + np.diag(cl, 1)

    # yl = np.linalg.solve(A, fl)
    
    
    if be0 != 0:
        yn = (ga1 - bn * yl[-1] - an * yl[-2]) / cn
        yl = np.append(yl, yn)
        xl = np.append(xl, r)
        
    if be1 != 0:
        y0 = (ga0 - b0 * yl[0] - c0 * yl[1]) / a0
        yl = np.append([y0], yl)
        xl = np.append([l], xl)
        
    return np.array( xl ), np.array( yl )
    

fig, (ax, ax2) = plt.subplots(1, 2)

n = 100
l, r = 0, 1

xl, yl = solver(l, r, n)
y_ex = exact(xl)
print(xl)
print(yl)
print(y_ex)


my_sol, = ax.plot(xl, yl, 'bo-')
exa, = ax.plot(xl, y_ex, 'g')
ax.legend(["TDMA", "Exact"])

diff = np.abs(yl - y_ex)
diff_p, = ax2.plot(xl, diff)
ax2.legend([np.max(diff)])


def upd(val_n):
    xl, yl = solver(l, r, val_n)
    y_ex = exact(xl)
    
    my_sol.set_data(xl, yl)
    exa.set_data(xl, y_ex)
    
    diff = np.abs(yl - y_ex)
    diff_p.set_data(xl, diff)
    ax2.legend([np.max(diff)])
    
n_ax = fig.add_axes([0.1, 0.02, 0.8, 0.03])
n_slider = Slider(
        ax=n_ax,
        label='n',
        valmin=3,
        valmax=1000,
        valinit=n,
        valstep=1
    )
n_slider.on_changed(upd)

plt.show()
