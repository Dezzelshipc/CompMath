import os
from math import *
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, RangeSlider, TextBox
import urllib.request, urllib.parse, re
import numpy as np
import plotly.graph_objs as go
from scipy.special import binom
from scipy.integrate import quad

import json


def request(req, additional="", formated='moutput'):
    data = {
        'input': req,
        'appid': wolfram_appid,
        'format': formated,
        'output': 'json',
    }
    url = 'http://api.wolframalpha.com/v1/query'

    url_values = urllib.parse.urlencode(data)

    req_url = "?".join([url, url_values]) + additional
    print(req)
    print(req_url)

    with urllib.request.urlopen(req_url) as resp_:
        return resp_.read().decode()


def min_max_parse(req):
    resp = request(req, "&includepodid=GlobalMinima&includepodid=GlobalMaxima")

    resp = re.findall('("moutput":".+")', resp)
    resp = [float(re.search('(\{.*, {)', out).group()
                  .replace(",", "").replace("{", "").replace("*^", "e")) for out in resp]
    return min(resp), max(resp)


def extr_d(n, str_func, l_bound, r_bound):
    string = f"Extrema[{{D[{str_func},{{x,{n + 1}}}],{l_bound}<=x<={r_bound}}},{{x}}]"
    return min_max_parse(string)


def extr_r(n, str_func, l_bound, r_bound, str_omega):
    string = f"Extrema[{{D[({str_func}){str_omega},{{x,{n + 1}}}]," \
             f" {l_bound}<=x<={r_bound},{l_bound}<=y<={r_bound}}},{{x,y}}]"
    return min_max_parse(string)


def L(n, x, tab: list):
    s = 0

    first, last = first_last(n, x, tab)

    for i in range(first, last):
        s += f(tab[i]) * prod([
            (x - tab[j]) / (tab[i] - tab[j]) if i != j else 1
            for j in range(first, last)
        ])
    return s


def erm_spline_coeffs(y1, y2, y1_, y2_, h):
    a = 6 / h * ((y2 - y1) / h - (2 * y1_ + y2_) / 3)
    b = 12 / (h * h) * ((y1_ + y2_) / 2 - (y2 - y1) / h)

    return a, b


def erm_spline(x, x1, y1, y2, y1_, y2_, h):
    a, b = erm_spline_coeffs(y1, y2, y1_, y2_, h)

    return y1 + y1_ * (x - x1) + a * (x - x1) ** 2 / 2 + b * (x - x1) ** 3 / 6


def erm_spline_d(x, x1, y1, y2, y1_, y2_, h):
    a, b = erm_spline_coeffs(y1, y2, y1_, y2_, h)

    return y1_ + a * (x - x1) + b * (x - x1) ** 2 / 2


def erm_spline_d2(x, x1, y1, y2, y1_, y2_, h):
    a, b = erm_spline_coeffs(y1, y2, y1_, y2_, h)

    return a + b * (x - x1)


def erm_spline_d3(x, x1, y1, y2, y1_, y2_, h):
    a, b = erm_spline_coeffs(y1, y2, y1_, y2_, h)

    return b


def erm_spline_dm1(x, x1, y1, y2, y1_, y2_, h):
    a, b = erm_spline_coeffs(y1, y2, y1_, y2_, h)

    return y1 * x + y1_ * (x - x1) ** 2 / 2 + a * (x - x1) ** 3 / 6 + b * (x - x1) ** 4 / 24


def L_c(n, x, tab):
    first, last = first_last(n, x, tab)

    coeffs = [0] * (n + 1)

    for j in range(first, last):
        c1 = 1
        c2 = [1]
        for i in range(first, last):
            if j != i:
                c1 *= tab[j] - tab[i]
                c2_1 = [-y * tab[i] for y in c2]

                c2.append(0)
                for k in range(len(c2) - 1):
                    c2[k + 1] += c2_1[k]

        for k in range(n + 1):
            coeffs[k] += c2[k] * f(tab[j]) / c1

    return np.around(coeffs, 6)


def derivative(coeffs):
    n = len(coeffs) - 1
    return [coeffs[i] * (n - i) for i in range(n)]


def derivative_n(n, coeffs):
    c = coeffs
    for _ in range(n):
        c = derivative(c)
    return c


def first_last(n, x, tab):
    start_i = lowerbound(tab, x)

    first, last = start_i - n // 2, start_i + (n + 1) // 2 + 1

    if first < 0:
        first = 0
        last = n + 1 if n + 1 <= len(tab) else len(tab)
    elif last > len(tab):
        last = len(tab)
        first = last - n - 1 if last - n - 1 >= 0 else 0
    return first, last


def lowerbound(lst, value):
    for i in range(len(lst)):
        if lst[i] > value:
            return i - 1
    return 0


class Plotter:
    degree = 1
    intervals_count = 10
    interval_division = 2
    begin = 0
    end = 10.0
    delta = 1
    start2 = 0
    end2 = 0

    coeff = 1

    fig, ax = (0, 0)

    nodes = []
    dots = []
    function_y = []
    patches = []

    nodes_plot = []
    inter_plot = []
    function_plot = []

    coeffs = []
    polynom_plot = []

    cur_func = []

    def min_max_on_plot(self):
        return min(self.function_y), max(self.function_y)

    def calc_nodes(self):
        self.nodes = np.linspace(self.start2, self.end2, num=self.intervals_count + 2 * (self.degree + 1) + 1)

    def calc_dots(self):
        precision = int(4 - log10(self.delta / self.interval_division))
        self.dots = np.around(np.linspace(self.begin, self.end, num=self.intervals_count * self.interval_division + 1),
                              precision)

    def additional_ends(self):
        self.delta = (self.end - self.begin) / self.intervals_count
        self.start2 = self.begin - self.delta * (self.degree + 1)
        self.end2 = self.end + self.delta * (self.degree + 1)

    def calc_f(self, func) -> np.array:
        return func(self.dots)

    def calc_f_dots(self, func):
        return func(self.nodes_f_plot())

    def nodes_f_plot(self) -> np.array:
        return self.nodes[self.degree + 1: -self.degree - 1]

    def calc_f_nodes(self, func):
        return func(self.nodes)

    def interpolate_l(self) -> list[float]:
        return [
            L(self.degree, x, self.nodes) for x in self.dots
        ]

    def interpolate_l2(self, n=0) -> list[float]:
        return [
            self.dP(L_c(self.degree, x, self.nodes), x, n) for x in self.dots
        ]

    def dP(self, coeffs, x, n):
        coeffs = derivative_n(n, coeffs)
        s = 0
        for xi in coeffs:
            s *= x
            s += xi
        return s

    def calc_erm_y(self, n=0):
        res = []
        func_y = self.calc_f_nodes(f)

        spline = ()
        match n:
            case -1:
                spline = erm_spline_dm1
            case 0:
                spline = erm_spline
            case 1:
                spline = erm_spline_d
            case 2:
                spline = erm_spline_d2
            case 3:
                spline = erm_spline_d3

        for x in self.dots:
            i = lowerbound(self.nodes, x)
            res.append(spline(x, self.nodes[i], func_y[i], func_y[i + 1],
                              self.df(self.nodes[i]), self.df(self.nodes[i + 1]), self.delta))
        return res

    def get_raw_function(self, n=0):
        t = np.arange(self.begin, self.end, 0.01)
        self.function_y = self.df(t, n)
        return t, self.function_y

    def f(self, x):
        # return x ** 2 - x * sin(x * self.coeff)
        return np.sin(x * self.coeff)
        # return x*2
        # return x ** 3 - sin(x)
        # return x - cos(x)
        # return sin(x)/x if x != 0 else 1

    def df(self, x, n=1):
        return (self.coeff ** n) * np.sin(self.coeff * x + pi * n / 2)

    def f_str(self):
        # return f"x**2 - x*sin({self.coeff}*x)"
        return f"sin({self.coeff} * x)"
        # return "x**3-sin(x)"

    def add_functionality(self):
        l_markersize = 2
        self.interpolated_y = self.interpolate_l2()

        _y_mi, _y_ma = self.min_max_on_plot()
        _y_diff = (_y_ma - _y_mi) / 100
        ni, = plt.plot(self.nodes_f_plot()[0:2], [_y_mi - _y_diff] * 2, 'co', linestyle='-', visible=False)

        # self.coeffs = L_c(self.degree,
        #                   sum(self.nodes[self.degree + 1:self.degree + 3]) / 2,
        #                   self.nodes)
        # p_x_ = np.linspace(self.begin, self.end, 1000)
        # p_y_ = [self.P(self.coeffs, x) for x in p_x_]
        # self.polynom_plot, = plt.plot(p_x_, p_y_, 'mo', linestyle='-', markersize=0)
        # print(self.coeffs)

        l, = plt.plot(self.dots, self.interpolated_y, 'ko', markersize=l_markersize, linestyle='-')
        self.spline_plot, = plt.plot(self.dots, self.calc_erm_y(), 'ro', markersize=l_markersize, linestyle='-')

        pn, = plt.plot(self.nodes_f_plot(), self.calc_f_dots(self.cur_func), 'go', markersize=3)

        def ax_get(n):
            sf = 0.17
            h = 0.025
            return sf - h * n

        axcolor = 'lightgoldenrodyellow'
        axint = plt.axes([0.25, ax_get(0), 0.65, 0.03], facecolor=axcolor)
        axdiv = plt.axes([0.25, ax_get(1), 0.65, 0.03], facecolor=axcolor)
        axdeg = plt.axes([0.25, ax_get(2), 0.65, 0.03], facecolor=axcolor)
        axrange = plt.axes([0.25, ax_get(3), 0.65, 0.03], facecolor=axcolor)
        axcoeff = plt.axes([0.25, ax_get(4), 0.65, 0.03], facecolor=axcolor)
        axnint = plt.axes([0.25, ax_get(5), 0.65, 0.03], facecolor=axcolor)

        sints = Slider(axint, 'Intervals', 1, 100, valinit=self.intervals_count, valstep=1, color='green')
        sdivs = Slider(axdiv, 'Int Divs', 1, 20, valinit=self.interval_division, valstep=1, color='lime')
        sdeg = Slider(axdeg, 'Degree', 1, 10, valinit=self.degree, valstep=1)
        srange = RangeSlider(axrange, 'Range', -20, 20, valinit=(self.begin, self.end))
        scoeff = Slider(axcoeff, 'Coeff', 0, 10, valinit=self.coeff, color='red')
        snint = Slider(axnint, 'Num Int', 0, self.intervals_count - 1, valinit=0, valstep=1, color='cyan')
        snint.ax.set_visible(False)

        def range_update(val=""):
            self.begin, self.end = srange.val
            self.coeff = scoeff.val

            t, ft = self.get_raw_function()
            self.function_plot.set_data(t, ft)

            x_padding = (self.end - self.begin) / 50
            self.ax.set_xlim([self.begin - x_padding, self.end + x_padding])

            y_min, y_max = self.min_max_on_plot()
            y_padding = (y_max - y_min + 0.0001) / 50
            self.ax.set_ylim([y_min - y_padding, y_max + y_padding])
            self.actual_value = quad(f, *srange.val)[0]

            update()

        self.stop = False

        def update(val=""):
            if self.stop:
                return

            self.stop = True
            self.degree = sdeg.val

            self.intervals_count = sints.val
            self.interval_division = sdivs.val
            sints.ax.set_xlim(sints.valmin, sints.valmax)
            sints.set_val(min(sints.val, sints.valmax))

            snint.valmax = self.intervals_count - 1
            snint.ax.set_xlim(snint.valmin, snint.valmax)
            snint.set_val(min(snint.val, snint.valmax))

            self.additional_ends()
            self.calc_nodes()
            self.calc_dots()

            self.stop = False
            change_mode(radio.value_selected)

        sints.on_changed(update)
        sdivs.on_changed(update)
        sdeg.on_changed(update)
        srange.on_changed(range_update)
        scoeff.on_changed(range_update)
        snint.on_changed(update)

        rax = plt.axes([0.025, 0.65, 0.15, 0.3], facecolor=axcolor)
        radio_text = ['F', 'dF', 'd2F', 'd3F', 'd-1F']
        dn_numbers = [0, 1, 2, 3, -1]
        radio = RadioButtons(rax, radio_text, active=0)

        def change_mode(val):
            if self.stop:
                return

            self.stop = True

            dn = dn_numbers[radio_text.index(val)]

            pn.set_data(self.nodes_f_plot(), self.calc_f_dots(self.cur_func))
            y_mi, y_ma = self.min_max_on_plot()
            y_diff = (y_ma - y_mi) / 100
            ni.set_data(self.nodes_f_plot()[snint.val:snint.val + 2], [y_mi - y_diff] * 2)
            self.spline_plot.set_data(self.dots, self.calc_erm_y(dn))
            l.set_data(self.dots, self.interpolate_l2(dn))
            self.function_plot.set_data(*self.get_raw_function(dn))

            self.stop = False
            self.fig.canvas.draw_idle()

        radio.on_clicked(change_mode)

        check_labels = ["Toggle function", "Toggle nodes", "Show Interval"]
        check_bools = [True, True, False]

        cax = plt.axes([0.025, 0.48, 0.15, 0.15], facecolor=axcolor)
        check = CheckButtons(cax, check_labels, check_bools)

        def turn_checks(val):
            if val == check_labels[0]:
                self.function_plot.set_visible(not self.function_plot.get_visible())
            elif val == check_labels[1]:
                self.spline_plot.set_markersize(0 if pn.get_visible() else l_markersize)
                l.set_markersize(0 if pn.get_visible() else l_markersize)
                pn.set_visible(not pn.get_visible())
            elif val == check_labels[2]:
                ni.set_visible(not ni.get_visible())
                snint.ax.set_visible(ni.get_visible())
            self.fig.canvas.draw_idle()

        check.on_clicked(turn_checks)

        plt.show()

    @staticmethod
    def set_coordinates():
        ax = plt.gca()
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_position('zero')
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')

    def table(self, names, vals):
        table = go.Table(
            header=dict(values=names),
            cells=dict(values=vals)
        )
        fig = go.Figure(data=table)
        fig.show()

    def start(self):
        self.cur_func = self.f
        self.fig, self.ax = plt.subplots()

        plt.subplots_adjust(left=0.25, bottom=0.25, top=0.99, right=0.99)

        self.additional_ends()

        self.calc_nodes()
        self.calc_dots()

        t, ft = self.get_raw_function()
        self.function_plot, = plt.plot(t, ft, 'bo', markersize=0, linestyle='-', linewidth=2)

        # self.set_coordinates()

        self.add_functionality()


# write here wolfram api appid for full data
wolfram_appid = os.environ.get('WOLFRAM_APPID', "")

f = ()

if __name__ == "__main__":
    p = Plotter()
    f = p.f
    p.start()
