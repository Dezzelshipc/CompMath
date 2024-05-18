import numpy as np

def right_force_firc(t, ab):
    return np.array([
        ab[1],
        Af * np.sin(wf * t) - k * ab[1] - w**2 * ab[0]
    ])

def model(y0, right):
    x_, y_ = runge_kutta(right, y0, t0, tn, (tn-t0)/n)
    y_ = y_.T
    
    return x_, y_

t0, tn = 0, 1000
n = 10000

w = np.sqrt( 9.8 / 1 )
л = 0.01
Af = 1
wf = 0.5

init = [np.pi/10, 0]
wfs = np.linspace(-1, 1, 41)**5
wfs = 0.1 * wfs + w
for k in [0.1, 0.05, 0.01]:
    Al = []
    for wfi in wfs:
        wf = wfi
        x_, y_ = model(init, right_force_firc)
        Al.append(max(abs( y_[0][-min(n//4, 100):] )))

    wfs, Al