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



# Var 5
al0, be0, ga0 = 1, -2, 1
al1, be1, ga1 = 1, 0, 0.5

# Var 1
# al0, be0, ga0 = 1, 0, 1
# al1, be1, ga1 = 1, 0, 0.5

# Var 8
# al0, be0, ga0 = 0, 1, 0.5
# al1, be1, ga1 = 1, 0, np.sqrt(2)

# Var 9
# al0, be0, ga0 = 3, -1, 1
# al1, be1, ga1 = 0, 1, np.sqrt(2)

def exact(x):
    return 1 / (x ** 2 + 1)
    # return 1 / (x+1)
    # return np.sqrt(x+1)
    # return 2/3 * (x+1)**(3/2)

def p(x):
    return -(x ** 2 + 1)
    # return -x+1
    # return 1/(2*(x+1))
    # return 0

def q(x):
    return -2*x
    # return -1
    # return -1
    # return -3/(x+1)**2

def f(x):
    return 2 * (3 * x**2 - 1) / (x ** 2 + 1) ** 3
    # return -2*x/(x+1)**3
    # return -np.sqrt(x+1)
    # return -3/(2 * np.sqrt(x+1))

def ac(h, p_):
    r = p_ * h / 2
    mid = abs(r) + np.exp(-abs(r))
    # mid = 1
    # mid = np.arctan(r)/r
    # mid = 1 + r**2 / (1 + abs(r) + np.sin(r)**2)
    return (mid - r) / h ** 2, (mid + r) / h ** 2


def solver(l, r, n):
    al, bl, cl, fl = [], [], [], []
    al.append(0)
    xl = np.linspace(l, r, n+1)
    
    h = (r - l) / n
    
    x = l
    i_c = 1

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
        k = a1/a0
        
        bl.append(b1 - k * b0)
        cl.append(c1 - k * c0)
        fl.append(f(x) - k * ga0)
        i_c = 2


    end_i = n-1 if be1 != 0 else n

    for i in range(i_c, end_i):
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
        
        k = cn1 / cn
        
        al.append(an1 - k * an)
        bl.append(bn1 - k * bn)
        fl.append(f(x) - k * ga1)
    
    # tf = np.array([abs(al[i])+abs(cl[i]) <= abs(bl[i]) for i in range(1,len(cl))])
    # print(tf.all())

    cl.append(0)
    yl = TDMA(al, bl, cl, fl)
    # print(xl, al, bl, cl, fl, sep='\n')
    
    
    # A = np.diag(al, -1) + np.diag(bl) + np.diag(cl, 1)

    # yl = np.linalg.solve(A, fl)
    
    if be0 != 0:
        y0 = (ga0 - b0 * yl[0] - c0 * yl[1]) / a0
        yl = np.append([y0], yl)
        
    if be1 != 0:
        yn = (ga1 - bn * yl[-1] - an * yl[-2]) / cn
        yl = np.append(yl, yn)
    

    return np.array( xl ), np.array( yl )
    

fig, (ax, ax2) = plt.subplots(1, 2)

n = 100
l, r = 0, 1

xl, yl = solver(l, r, n)
y_ex = exact(xl)

# np.set_printoptions(formatter={'all': lambda x: "{:.4g}".format(x)})
# print(yl)
# print(y_ex)

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
    
    ax2.relim()
    ax2.autoscale_view()
    
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
