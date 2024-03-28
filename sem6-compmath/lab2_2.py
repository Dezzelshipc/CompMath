import numpy as np
import math
from enum import Enum
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

def TDMA(a,b,c,f):
    a, b, c, f = tuple(map(lambda k_list: list(map(float, k_list)), (a, c, b, f)))

    alpha = [-b[0] / c[0]]
    beta = [f[0] / c[0]]
    n = len(f)
    x = [0]*n

    for i in range(1, n):
        alpha.append(-b[i]/(a[i]*alpha[i-1] + c[i]))
        beta.append((f[i] - a[i]*beta[i-1])/(a[i]*alpha[i-1] + c[i]))

    x[n-1] = beta[n - 1]

    for i in range(n-1, -1, -1):
        x[i - 1] = alpha[i - 1]*x[i] + beta[i - 1]

    return x


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
# al0, be0, ga0 = 1, -2, 1
# al1, be1, ga1 = 1, 0, 0.5

# Var 1
# al0, be0, ga0 = 1, 0, 1
# al1, be1, ga1 = 1, 0, 0.5

# Var 8
# al0, be0, ga0 = 0, 1, 0.5
# al1, be1, ga1 = 1, 0, np.sqrt(2)

# Var 9
al0, be0, ga0 = 3, -1, 1
al1, be1, ga1 = 0, 1, np.sqrt(2)

def exact(x):
    # return 1 / (x ** 2 + 1)
    # return 1 / (x+1)
    # return np.sqrt(x+1)
    return 2/3 * (x+1)**(3/2)

def p(x):
    # return -(x ** 2 + 1)
    # return -x+1
    # return 1/(2*(x+1))
    return 0

def q(x):
    # return -2*x
    # return -1
    # return -1
    return -3/(x+1)**2

def f(x):
    # return 2 * (3 * x**2 - 1) / (x ** 2 + 1) ** 3
    # return -2*x/(x+1)**3
    # return -np.sqrt(x+1)
    return -3/(2 * np.sqrt(x+1))

def ac(h, p_):
    r = p_ * h / 2
    # mid = abs(r) + np.exp(-abs(r))
    # mid = 1
    # mid = np.arctan(r)/r
    mid = 1 + r**2 / (1 + abs(r) + np.sin(r)**2)
    return (mid - r) / h ** 2, (mid + r) / h ** 2


def solver(l, r, n):
    al, bl, cl, fl = [], [], [], []
    al.append(0)
    xl = np.linspace(l, r, n+1)
    
    h = (r - l)/(n)

    x = l

    a0 = al0 - 3 * be0 / (2 * h)
    b0 = 2 * be0 / h
    c0 = -be0 / (2 * h)
    if be0 == 0:
        bl.append(al0)
        cl.append(0)
        fl.append(ga0)
    else:
        x = xl[1]
        a1, c1 = ac(h, p(x))
        
        b1 = q(x) - a1 - c1
        k = c0/c1
        
        bl.append(a0 - k * a1)
        cl.append(b0 - k * b1)
        fl.append(ga0 - k * f(x))


    end_i = n-1 if be0 != 0 else n

    for i in range(1, n):
        x = xl[i]
        
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
        al.append(0)
        bl.append(al1)
        fl.append(ga1)

    else:
        x = xl[-2]
        
        an1, cn1 = ac(h, p(x))
        bn1 = q(x) - an1 - cn1
        
        k = an1 / an
        
        x = xl[-1]
        al.append(0)
        bl.append(bn1 - k * bn)
        cl.append(cn1 - k * cn)
        fl.append(fl[-1] - k * ga1)
    
    # tf = np.array([abs(al[i])+abs(cl[i]) <= abs(bl[i]) for i in range(1,len(cl))])
    # print(tf.all())

    cl.append(0)
    yl = TDMA(al, bl, cl, fl)
    # print(xl, al, bl, cl, fl, sep='\n')
    
    
    # A = np.diag(al, -1) + np.diag(bl) + np.diag(cl, 1)

    # yl = np.linalg.solve(A, fl)
    
     
    # if be0 != 0:
    #     y0 = (ga0 - b0 * yl[0] - c0 * yl[1]) / a0
    #     yl = np.append([y0], yl)
    #     xl = np.append([l], xl)
        
    # if be1 != 0:
    #     yn = (ga1 - bn * yl[-1] - an * yl[-2]) / cn
    #     yl = np.append(yl, yn)
    #     xl = np.append(xl, r)

    return np.array( xl ), np.array( yl )
    

fig, (ax, ax2) = plt.subplots(1, 2)

n = 100
l, r = 0, 1

xl, yl = solver(l, r, n)
y_ex = exact(xl)

my_sol, = ax.plot(xl, yl, 'bo-', markersize=3)
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
