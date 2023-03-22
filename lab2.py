import os
from math import *
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons, CheckButtons, RangeSlider
import urllib.request, urllib.parse, re
import numpy as np
import plotly.graph_objs as go
from scipy.special import binom


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
    string = f"Extrema[{{D[{str_func},{{x,{n + 1}}}],{l_bound}<=x<={r_bound}}},{{x}}]"
    return request(string)


def extr_r(n, str_func, l_bound, r_bound, str_omega):
    string = f"Extrema[{{D[({str_func}){str_omega},{{x,{n + 1}}}]," \
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


def omega_str2(tab, left, right):
    return "".join([f"(y-{tab[i]})" for i in range(left, right)])


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


def N_rav1(n, x, tab, fin_diff, delta):
    k_l = lowerbound(tab, x)

    t = (tab[k_l] - x)/delta

    for i in range(n+1):
        print(len(fin_diff[i]), end=' ')
        print(fin_diff[i][k_l], end=' ')
    print(k_l, n+1, len(fin_diff))

    return sum(
        binom(t, i) * fin_diff[i][k_l] for i in range(n + 1)
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
    begin = 1.0
    end = 10.0
    delta = 1
    start2 = 0
    end2 = 0

    coeff = 2

    fig, ax = (0, 0)

    diff_type = 0

    finite_differences = []

    nodes = []
    dots = []
    function_y = []
    interpolated_y = []

    nodes_plot = []
    inter_plot = []
    function_plot = []

    def min_max_on_plot(self):
        return min(self.function_y), max(self.function_y)

    def calc_nodes(self):
        self.nodes = np.linspace(self.start2, self.end2, num=self.intervals_count + 2 * (self.degree + 1) + 1)

    def calc_dots(self):
        precision = int(4 - log10(self.delta / self.interval_division))
        self.dots = np.around(np.linspace(self.begin, self.end, num=self.intervals_count * self.interval_division + 1),
                              precision)

    def calc_fin_diff(self):
        match self.diff_type:
            case 0:
                self.calc_fin_diff_n_1()

    def calc_fin_f(self):
        match self.diff_type:
            case 0:
                return [
                    N_rav1(self.degree, x, self.nodes, self.finite_differences, self.delta) for x in self.dots
                ]

    def calc_fin_diff_n_1(self):
        self.finite_differences = [[]] * (self.degree + 1)
        self.finite_differences[0] = [f(x) for x in self.nodes]
        for i in range(1, self.degree + 1):
            cur = self.finite_differences[i - 1]
            self.finite_differences[i] = [cur[k + 1] - cur[k] for k in range(len(cur) - 1)]

    def additional_ends(self):
        self.delta = (self.end - self.begin) / self.intervals_count
        self.start2 = self.begin - self.delta * (self.degree + 1)
        self.end2 = self.end + self.delta * (self.degree + 1)

    def calc_f(self) -> list[float]:
        return [
            f(x) for x in self.dots
        ]

    def interpolate_l(self) -> list[float]:
        return [
            L(self.degree, x, self.nodes) for x in self.dots
        ]

    def calc_f_dots(self):
        return [f(x) for x in self.nodes_f_plot()]

    def nodes_f_plot(self) -> list[float]:
        return self.nodes[self.degree + 1: -self.degree - 1]

    def interpolate_n(self):
        return [
            N(self.degree, x, self.nodes) for x in self.dots
        ]

    def get_raw_function(self):
        t = np.arange(self.begin, self.end, 0.01)
        self.function_y = [f(k) for k in t]
        return t, self.function_y

    def f(self, x):
        return x ** 2 - x * sin(x * self.coeff)

    def f_str(self):
        return f"x**2 - x*sin({self.coeff}*x)"

    def add_functionality(self):
        l_markersize = 2
        self.interpolated_y = self.interpolate_l()

        l, = plt.plot(self.dots, self.interpolated_y, 'ro', markersize=l_markersize, linestyle='-')
        pn, = plt.plot(self.nodes_f_plot(), self.calc_f_dots(), 'go', markersize=3)

        _y_mi, _y_ma = self.min_max_on_plot()
        _y_diff = (_y_ma - _y_mi) / 100
        ni, = plt.plot(self.nodes_f_plot()[0:2], [_y_mi - _y_diff] * 2, 'co', linestyle='-')

        def ax_get(n):
            n_sliders = 6
            sf = 0.18
            se = 0.03
            return se + (sf - se) * (n_sliders - n) / n_sliders

        axcolor = 'lightgoldenrodyellow'
        axint = plt.axes([0.25, ax_get(0), 0.65, 0.03], facecolor=axcolor)
        axdiv = plt.axes([0.25, ax_get(1), 0.65, 0.03], facecolor=axcolor)
        axdeg = plt.axes([0.25, ax_get(2), 0.65, 0.03], facecolor=axcolor)
        axrange = plt.axes([0.25, ax_get(3), 0.65, 0.03], facecolor=axcolor)
        axcoeff = plt.axes([0.25, ax_get(4), 0.65, 0.03], facecolor=axcolor)
        axnint = plt.axes([0.25, ax_get(5), 0.65, 0.03], facecolor=axcolor)

        sints = Slider(axint, 'Intervals', 1, 50, valinit=self.intervals_count, valstep=1)
        sdivs = Slider(axdiv, 'Int Divs', 1, 20, valinit=self.interval_division, valstep=1)
        sdeg = Slider(axdeg, 'Degree', 1, 10, valinit=self.degree, valstep=1)
        srange = RangeSlider(axrange, 'Range', -20, 20, valinit=(self.begin, self.end))
        scoeff = Slider(axcoeff, 'Coeff', 0, 10, valinit=self.coeff)
        snint = Slider(axnint, 'Num Int', 0, self.intervals_count - 1, valinit=0, valstep=1)

        def range_update(val):
            self.begin, self.end = srange.val
            self.coeff = scoeff.val
            t, ft = self.get_raw_function()
            self.function_plot.set_data(t, ft)

            x_padding = (self.end - self.begin) / 50
            self.ax.set_xlim([self.begin - x_padding, self.end + x_padding])

            y_min, y_max = self.min_max_on_plot()
            y_padding = (y_max - y_min + 0.0001) / 50
            self.ax.set_ylim([y_min - y_padding, y_max + y_padding])

            update()

        self.stop = False

        def update(val=""):
            if self.stop:
                return

            self.stop = True
            if self.degree != sdeg.val:
                self.degree = sdeg.val
                self.calc_fin_diff_n_1()

            self.intervals_count = sints.val
            self.interval_division = sdivs.val

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

        rax = plt.axes([0.025, 0.5, 0.15, 0.15], facecolor=axcolor)
        radio = RadioButtons(rax, ['Lagrange', 'Newton', 'NewtR1', 'F'], active=0)

        def change_mode(val):
            match val:
                case 'Lagrange':
                    self.interpolated_y = self.interpolate_l()
                case 'Newton':
                    self.interpolated_y = self.interpolate_n()
                case 'NewtR1':
                    print(self.calc_fin_f())
                    self.interpolated_y = self.calc_fin_f()
                case _:
                    self.interpolated_y = self.calc_f()

            l.set_data(self.dots, self.interpolated_y)
            pn.set_data(self.nodes_f_plot(), self.calc_f_dots())
            y_mi, y_ma = self.min_max_on_plot()
            y_diff = (y_ma - y_mi) / 100
            ni.set_data(self.nodes_f_plot()[snint.val:snint.val + 2], [y_mi - y_diff] * 2)
            self.fig.canvas.draw_idle()

        radio.on_clicked(change_mode)

        check_labels = ["Toggle function", "Toggle nodes", "Show Interval"]
        check_bools = [True, True, True]

        cax = plt.axes([0.025, 0.3, 0.15, 0.15], facecolor=axcolor)
        check = CheckButtons(cax, check_labels, check_bools)

        def turn_checks(val):
            if val == check_labels[0]:
                self.function_plot.set_visible(not self.function_plot.get_visible())
            elif val == check_labels[1]:
                l.set_markersize(0 if pn.get_visible() else l_markersize)
                pn.set_visible(not pn.get_visible())
            elif val == check_labels[2]:
                ni.set_visible(not ni.get_visible())
            self.fig.canvas.draw_idle()

        check.on_clicked(turn_checks)

        axb = plt.axes([0.025, 0.15, 0.1, 0.05], facecolor=axcolor)
        button = Button(axb, "Show table")

        def show_table(val):
            data = [[]] * 6
            data[0] = self.dots
            data[1] = self.calc_f()
            data[2] = self.interpolate_l()
            data[3] = self.interpolate_n()
            data[4] = [data[1][i] - data[2][i] for i in range(len(self.dots))]
            data[5] = [data[2][i] - data[3][i] for i in range(len(self.dots))]

            self.table(('Dot', 'Func', 'Lagrange', 'Newton', 'R = F - L', 'L - N'), data)

        button.on_clicked(show_table)

        axb2 = plt.axes([0.025, 0.09, 0.1, 0.05], facecolor=axcolor)
        button2 = Button(axb2, "Show Rem Table")

        def show_rem_table(val):
            table = [[]] * 7

            table[0] = self.dots[
                       snint.val * self.interval_division: (snint.val + 1) * self.interval_division + 1]

            mi, ma = extr_r(self.degree,
                            self.f_str(),
                            self.dots[snint.val * self.interval_division],
                            self.dots[(snint.val + 1) * self.interval_division],
                            omega_str2(self.nodes, snint.val,
                                       snint.val + 1))

            table[4] = [mi] * (self.interval_division + 1)
            table[5] = [ma] * (self.interval_division + 1)

            table[1] = self.calc_f()[
                       snint.val * self.interval_division: (snint.val + 1) * self.interval_division + 1]
            table[2] = self.interpolate_l()[
                       snint.val * self.interval_division: (snint.val + 1) * self.interval_division + 1]
            table[3] = [table[1][i] - table[2][i] for i in range(len(table[0]))]

            table[6] = [abs(table[3][i]) < abs(table[5][i]) for i in range(len(table[0]))]

            self.table(('Dot', 'Func', 'Lagrange', 'R = F - L', 'R min', 'R max', 'Compare'), table)

        button2.on_clicked(show_rem_table)

        axb3 = plt.axes([0.025, 0.03, 0.1, 0.05], facecolor=axcolor)
        button3 = Button(axb3, "Show Diff Table")

        def show_diff_table(val):
            print(self.finite_differences)
            self.table([i for i in range(self.degree)], self.finite_differences)

        button3.on_clicked(show_diff_table)

        plt.show()

    def table(self, names, vals):
        table = go.Table(
            header=dict(values=names),
            cells=dict(values=vals)
        )
        fig = go.Figure(data=table)
        fig.show()

    def start(self):
        self.fig, self.ax = plt.subplots()

        plt.subplots_adjust(left=0.25, bottom=0.25, top=0.99, right=0.99)

        self.additional_ends()

        self.calc_nodes()
        self.calc_dots()

        t, ft = self.get_raw_function()
        self.function_plot, = plt.plot(t, ft, 'bo', markersize=0, linestyle='-', linewidth=2)

        self.calc_fin_diff()

        self.add_functionality()


# write here wolfram api appid for full data
wolfram_appid = os.environ.get('WOLFRAM_APPID', "")

f = ()

if __name__ == "__main__":
    p = Plotter()
    f = p.f
    p.start()
