import matplotlib.pyplot as plt
import numpy as np

# 0, 2
def u1(r, f):
    return 7/5 - 1/6 * np.sin(f) - 1/48 * np.cos(4 * f)

# 4, 5
def u2(r,f):
    return (3 * np.log(5) - 10 * np.log(4))/(2*(np.log(5) - np.log(4))) + 7/(2*(np.log(5) - np.log(4))) * np.log(r) - 23/18 *r*np.sin(f) + 184/(9 * r) * np.sin(f)

# 1.5+
def u3(r, f):
    return 1.5**7 / (12 * r**6) * np.cos(6 *f)

def u4(r, f):
    return 1 + (2/np.log(5))*np.log(r) + (2025 / r**2 + 25 * r**2) / 706 * np.sin(2*f) + (243 * r**4 - 94921875 /r**4) / 1588744 * np.cos(4*f)


rl = np.linspace(3, 5, 201)
fl = np.linspace(0, 2 * np.pi, 201)

rr, ff = np.meshgrid(rl, fl)

uu = u4(rr, ff)

fig = plt.figure(figsize=[5,5])
ax = fig.add_axes([0.1,0.1,0.8,0.8], polar=True)
ax.set_rlim(2.9, 5)

ax.pcolor(ff, rr, uu, rasterized=True)

plt.savefig('./sem6-mathpys/lab4(3)/plot.pdf')
plt.show()