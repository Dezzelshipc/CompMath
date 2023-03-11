import os
from math import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, RangeSlider
import urllib.request, urllib.parse, re
import numpy as np
import plotly.graph_objs as go


def request(req):
    data = {
        'input': req,
        'appid': wolfram_appid,
        'format': 'moutput',
        'output': 'json',
        'scanner': 'MaxMin',
    }
    url = 'http://api.wolframalpha.com/v1/query'

    url_values = urllib.parse.urlencode(data)

    req_url = "?".join([url, url_values]) + "&includepodid=GlobalMinima&includepodid=GlobalMaxima"
    print(req)
    print(req_url)

    with urllib.request.urlopen(req_url) as resp_:
        resp = resp_.read().decode()

        resp = re.findall('("moutput":".+")', resp)
        resp = [float(re.search('(\{.*, {)', out).group()
                      .replace(",", "").replace("{", "").replace("*^", "e")) for out in resp]
        return min(resp), max(resp)


def extr_d(n, str_func, l_bound, r_bound):
    string = f"Extrema [{{D[{str_func},{{x,{n + 1}}}],{l_bound}<=x<={r_bound}}},{{x}}]"
    return request(string)


def extr_r(n, str_func, l_bound, r_bound, str_omega):
    string = f"Extrema [{{D[({str_func}){str_omega},{{x,{n + 1}}}]," \
             f" {l_bound}<=x<={r_bound},{l_bound}<=y<={r_bound}}},{{x,y}}]"
    return request(string)


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


def omega(n, x, tab):
    return prod(x - tab[i] for i in range(*first_last(n, x, tab)))


def omega_str(n, x, tab):
    return "".join([f"(y-{tab[i]})" for i in range(*first_last(n, x, tab))])


def L(n, x, tab: list):
    s = 0

    first, last = first_last(n, x, tab)

    for i in range(first, last):
        s += f(tab[i]) * prod([
            (x - tab[j]) / (tab[i] - tab[j]) if i != j else 1
            for j in range(first, last)
        ])
    return s


def R(n, diff: tuple, om):
    fact = factorial(n + 1)
    rs = [diff[0] * om / fact, diff[1] * om / fact]
    return min(rs), max(rs)


def div_diff(first, last, tab):
    return sum(
        f(tab[j]) / prod(
            (tab[j] - tab[i]) if i != j else 1
            for i in range(first, last)
        )
        for j in range(first, last)
    )


def N(n, x, tab):
    first, last = first_last(n, x, tab)

    return sum(
        div_diff(first, i + 1, tab) * prod(
            x - tab[j]
            for j in range(first, i)
        ) for i in range(first, last)
    )


def lowerbound(lst, value):
    for i in range(len(lst)):
        if lst[i] > value:
            return i - 1
    return 0


class Plotter:
    degree = 1
    intervals_count = 10
    interval_division = 2
    start = 1.0
    end = 10.0
    delta = 1
    start2 = 0
    end2 = 0

    coeff = 2

    fig, ax = (0, 0)

    nodes = []
    dots = []

    nodes_plot = []
    inter_plot = []
    function_plot = []

    def calc_nodes(self):
        self.nodes = [round(self.start2 + i * self.delta, floor(log10(self.intervals_count) + 2)) for i in
                      range(self.intervals_count + 1 + 2 * (self.degree + 1))]

    def calc_dots(self):
        delta_ = self.delta / self.interval_division
        self.dots = [round(self.start + i * delta_, -floor(log10(delta_)) + 1) for i in
                     range(self.intervals_count * self.interval_division + 1)]

    def additional_ends(self):
        self.delta = (self.end - self.start) / self.intervals_count
        self.start2 = self.start - self.delta * (self.degree + 1)
        self.end2 = self.end + self.delta * (self.degree + 1)

    def calc_f(self) -> list[float]:
        return [
            f(x) for x in self.dots
        ]

    def interpolate_l(self) -> list[float]:
        return [
            L(self.degree, x, self.nodes) for x in self.dots
        ]

    def calc_f_nodes(self) -> list[float]:
        return [f(x) for x in self.nodes][self.degree + 1: -self.degree - 1]

    def nodes_f_plot(self) -> list[float]:
        return self.nodes[self.degree + 1: -self.degree - 1]

    def interpolate_n(self):
        return [
            N(self.degree, x, self.nodes) for x in self.dots
        ]

    def get_raw_function(self):
        t = np.arange(self.start, self.end, 0.01)
        ft = [f(k) for k in t]
        return t, ft

    def f(self, x):
        return sin(x * self.coeff)

    def add_functionality(self):
        l_markersize = 2

        l, = plt.plot(self.dots, self.interpolate_l(), 'ro', markersize=l_markersize, linestyle='-')
        pn, = plt.plot(self.nodes_f_plot(), self.calc_f_nodes(), 'go', markersize=3)

        axcolor = 'lightgoldenrodyellow'
        axint = plt.axes([0.25, 0.15, 0.65, 0.03], facecolor=axcolor)
        axdiv = plt.axes([0.25, 0.12, 0.65, 0.03], facecolor=axcolor)
        axdeg = plt.axes([0.25, 0.09, 0.65, 0.03], facecolor=axcolor)
        axrange = plt.axes([0.25, 0.06, 0.65, 0.03], facecolor=axcolor)
        axcoeff = plt.axes([0.25, 0.03, 0.65, 0.03], facecolor=axcolor)

        sints = Slider(axint, 'Intervals', 1, 50, valinit=self.intervals_count, valstep=1)
        sdivs = Slider(axdiv, 'Int Divs', 1, 20, valinit=self.interval_division, valstep=1)
        sdeg = Slider(axdeg, 'Degree', 1, 10, valinit=self.degree, valstep=1)
        srange = RangeSlider(axrange, 'Range', -20, 20, valinit=(self.start, self.end))
        scoeff = Slider(axcoeff, 'Coeff', 0, 10, valinit=self.coeff)

        def range_update(val):
            self.start, self.end = srange.val
            self.coeff = scoeff.val
            t, ft = self.get_raw_function()
            self.function_plot.set_data(t, ft)

            padding = (self.end - self.start) / 50
            self.ax.set_xlim([self.start - padding, self.end + padding])
            update()

        def update(val=""):
            self.degree = sdeg.val
            self.intervals_count = sints.val
            self.interval_division = sdivs.val
            self.additional_ends()
            self.calc_nodes()
            self.calc_dots()
            change_mode(radio.value_selected)

        sints.on_changed(update)
        sdivs.on_changed(update)
        sdeg.on_changed(update)
        srange.on_changed(range_update)
        scoeff.on_changed(range_update)

        rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
        radio = RadioButtons(rax, ['Lagrange', 'Newton', 'F'], active=0)

        def change_mode(val):
            tab = []
            if val == 'Lagrange':
                tab = self.interpolate_l()
            elif val == 'Newton':
                tab = self.interpolate_n()
            else:
                tab = self.calc_f()
            l.set_data(self.dots, tab)
            pn.set_data(self.nodes_f_plot(), self.calc_f_nodes())
            self.fig.canvas.draw_idle()

        radio.on_clicked(change_mode)

        check_labels = ["Toggle function", "Toggle nodes"]
        check_bools = [True, True]

        cax = plt.axes([0.025, 0.3, 0.15, 0.15], facecolor=axcolor)
        check = CheckButtons(cax, check_labels, check_bools)

        def turn_checks(val):
            if val == check_labels[0]:
                self.function_plot.set_visible(not self.function_plot.get_visible())
            elif val == check_labels[1]:
                l.set_markersize(0 if pn.get_visible() else l_markersize)
                pn.set_visible(not pn.get_visible())
            self.fig.canvas.draw_idle()

        check.on_clicked(turn_checks)

        axb = plt.axes([0.025, 0.1, 0.1, 0.1], facecolor=axcolor)
        button = Button(axb, "Show table")

        def show_table(val):
            self.table()

        button.on_clicked(show_table)

        plt.show()

    def table(self):
        data = [[]]*6
        data[0] = self.dots
        data[1] = self.calc_f()
        data[2] = self.interpolate_l()
        data[3] = self.interpolate_n()
        data[4] = [data[1][i] - data[2][i] for i in range(len(self.dots))]
        data[5] = [data[2][i] - data[3][i] for i in range(len(self.dots))]

        table = go.Table(
            header=dict(values=('Dot', 'Func', 'Lagrange', 'Newton', 'R = F - L', 'L - N')),
            cells=dict(values=data)
        )
        fig = go.Figure(data=table)
        fig.show()

    def plot(self):
        self.fig, self.ax = plt.subplots()

        plt.subplots_adjust(left=0.25, bottom=0.25, top=0.99, right=0.99)

        self.additional_ends()

        self.calc_nodes()
        self.calc_dots()

        t, ft = self.get_raw_function()
        self.function_plot, = plt.plot(t, ft, 'bo', markersize=0, linestyle='-', linewidth=2)

        self.add_functionality()


# write here wolfram api appid for full data
wolfram_appid = ""# os.environ.get('WOLFRAM_APPID')

f = ()

if __name__ == "__main__":
    p = Plotter()
    f = p.f
    p.plot()
