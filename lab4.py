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


def integral_parse(req):
    resp = request(req, "&includepodid=Input&podstate=Input__More+digits")

    resp = json.loads(resp)["queryresult"]["pods"][0]["subpods"][0]["moutput"]
    delim = "≈" if "≈" in resp else "="
    pos_delim = resp.rfind(delim)
    return float(resp[pos_delim + 1:])


def diff_parse(req):
    # resp = request(req, "&includepodid=AlternateForm")

    resp = """{
	"queryresult":{
		"success":true,
		"error":false,
		"numpods":1,
		"datatypes":"",
		"timedout":"",
		"timedoutpods":"",
		"timing":1.025,
		"parsetiming":0.552,
		"parsetimedout":false,
		"recalculate":"",
		"id":"MSP514412b75127dig4384c000068fb415a3d9ab72f",
		"host":"https:\/\/www6b3.wolframalpha.com",
		"server":"1",
		"related":"https:\/\/www6b3.wolframalpha.com\/api\/v1\/relatedQueries.jsp?id=MSPa514512b75127dig4384c00002id8e7f27h10g91e7794153569472399936",
		"version":"2.6",
		"inputstring":"D[sin(x)(y-1)(y-2),{x,2},{y,1}]",
		"pods":[
			{
				"title":"Alternate forms",
				"scanner":"Simplification",
				"id":"AlternateForm",
				"position":100,
				"error":false,
				"numsubpods":3,
				"subpods":[
					{
						"title":"",
						"moutput":"(3 - 2 y) Sin[x]"
					},
					{
						"title":"",
						"moutput":"-((-3 + 2 y) Sin[x])"
					},
					{
						"title":"",
						"moutput":"(3 I)\/2 E^(-I x) - (3 I)\/2 E^(I x) - I E^(-I x) y + I E^(I x) y"
					}
				],
				"expressiontypes":[
					{
						"name":"Default"
					},
					{
						"name":"Default"
					},
					{
						"name":"Default"
					}
				]
			}
		]
	}
}"""
    resp = re.findall('("moutput":".+")', resp)
    return resp[0][11:-1].replace(" ", '')


def extr_d(n, str_func, l_bound, r_bound):
    string = f"Extrema[{{D[{str_func},{{x,{n + 1}}}],{l_bound}<=x<={r_bound}}},{{x}}]"
    return min_max_parse(string)


def extr_r(n, str_func, l_bound, r_bound, str_omega):
    string = f"Extrema[{{D[({str_func}){str_omega},{{x,{n + 1}}}]," \
             f" {l_bound}<=x<={r_bound},{l_bound}<=y<={r_bound}}},{{x,y}}]"
    return min_max_parse(string)


def diff_req(n, k, str_func, str_omega):
    string = f"D[({str_func}){str_omega},{{x,{n + 1}}},{{y,{k}}}]"
    return diff_parse(string)


def integral_req(str_func, left, right):
    string = f"Integrate[{str_func},{{x,{left},{right}}}]"
    return integral_parse(string)


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


def omega_coeffs(n, x, tab):
    first, last = first_last(n, x, tab)

    coeffs = [1]

    for i in range(first, last):
        c_1 = [-y * tab[i] for y in coeffs]

        coeffs.append(0)
        for k in range(len(coeffs) - 1):
            coeffs[k + 1] += c_1[k]

    return np.around(coeffs, 6)


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


def L_c(n, x, tab) -> list:
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

    t = (x - tab[k_l]) / delta

    return sum(
        binom(t, i) * fin_diff[i][k_l] for i in range(n + 1)
    )


def N_rav2(n, x, tab, fin_diff, delta):
    k_l = lowerbound(tab, x)

    t = (tab[k_l] - x) / delta

    return sum(
        binom(t + i - 1, i) * fin_diff[i][k_l - i - 1] for i in range(n + 1)
    )


def lowerbound(lst, value):
    for i in range(len(lst)):
        if lst[i] > value:
            return i - 1
    return 0


class Plotter:
    degree = 1
    intervals_count = 10
    intervals_mult = 2
    begin = 0
    end = 10.0
    delta = 1
    start2 = 0
    end2 = 0

    coeff = 1

    fig, ax = (0, 0)

    diff_type = -1

    calculated = 0
    actual_value = 1.83907

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
        precision = int(4 - log10(self.delta / self.intervals_mult))
        self.dots = np.around(np.linspace(self.begin, self.end, num=self.intervals_count * self.intervals_mult + 1),
                              precision)

    def additional_ends(self):
        self.delta = (self.end - self.begin) / self.intervals_count
        self.start2 = self.begin - self.delta * (self.degree + 1)
        self.end2 = self.end + self.delta * (self.degree + 1)

    def calc_f(self, func) -> list[float]:
        return [
            func(x) for x in self.dots
        ]

    def calc_f_dots(self, func):
        return [func(x) for x in self.nodes_f_plot()]

    def nodes_f_plot(self) -> list[float]:
        return self.nodes[self.degree + 1: -self.degree - 1]

    def calc_f_nodes(self, func):
        return [func(x) for x in self.nodes]

    def get_raw_function(self, func):
        t = np.arange(self.begin, self.end, 0.01)
        self.function_y = [func(k) for k in t]
        return t, self.function_y

    def f(self, x):
        # return x ** 2 - x * sin(x * self.coeff)
        return sin(x * self.coeff)
        # return x*2
        # return x ** 3 - sin(x)
        # return x - cos(x)

    def df(self, x, n=1):
        return (self.coeff ** n) * sin(self.coeff * x + pi * n / 2)

    def f_str(self):
        # return f"x**2 - x*sin({self.coeff}*x)"
        return f"sin({self.coeff} * x)"
        # return "x**3-sin(x)"

    @staticmethod
    def poly(x, y):
        return patches.Polygon(xy=list(zip(x, y)),
                               alpha=0.5,
                               edgecolor='red',
                               linewidth=1,
                               closed=False)

    def left_rectangles_poly(self):
        lst = []
        f_nodes = self.calc_f_nodes(f)
        for i in range(self.degree + 1, self.degree + self.intervals_count + 1):
            x = [self.nodes[i], self.nodes[i], self.nodes[i + 1], self.nodes[i + 1]]
            y = [0, f_nodes[i], f_nodes[i], 0]
            lst.append(self.poly(x, y))
        return lst

    def calc_left_rectangles(self):
        return sum(
            x for x in self.calc_f_nodes(f)[self.degree + 1: self.degree + self.intervals_count + 1]
        ) * self.delta

    def right_rectangles_poly(self):
        lst = []
        f_nodes = self.calc_f_nodes(f)
        for i in range(self.degree + 1, self.degree + self.intervals_count + 1):
            x = [self.nodes[i], self.nodes[i], self.nodes[i + 1], self.nodes[i + 1]]
            y = [0, f_nodes[i + 1], f_nodes[i + 1], 0]
            lst.append(self.poly(x, y))
        return lst

    def calc_right_rectangles(self):
        return sum(
            x for x in self.calc_f_nodes(f)[self.degree + 2: self.degree + self.intervals_count + 2]
        ) * self.delta

    def central_rectangles_poly(self):
        lst = []
        nodes = [(self.nodes[i] + self.nodes[i + 1]) / 2 for i in range(len(self.nodes) - 1)]
        f_nodes = [f(x) for x in nodes]
        for i in range(self.degree + 1, self.degree + self.intervals_count + 1):
            x = [self.nodes[i], self.nodes[i], self.nodes[i + 1], self.nodes[i + 1]]
            y = [0, f_nodes[i], f_nodes[i], 0]
            lst.append(self.poly(x, y))
        return lst

    def calc_central_rectangles(self):
        nodes = [(self.nodes[i] + self.nodes[i + 1]) / 2 for i in range(len(self.nodes) - 1)]
        f_nodes = [f(x) for x in nodes]
        return sum(
            x for x in f_nodes[self.degree + 1: self.degree + self.intervals_count + 1]
        ) * self.delta

    def trapezoid_poly(self):
        lst = []
        f_nodes = self.calc_f_nodes(f)
        for i in range(self.degree + 1, self.degree + self.intervals_count + 1):
            x = [self.nodes[i], self.nodes[i], self.nodes[i + 1], self.nodes[i + 1]]
            y = [0, f_nodes[i], f_nodes[i + 1], 0]
            lst.append(self.poly(x, y))
        return lst

    def calc_trapezoid(self):
        f_nodes = self.calc_f_nodes(f)
        return sum(
            (f_nodes[i] + f_nodes[i + 1]) / 2 for i in range(self.degree + 1, self.degree + self.intervals_count + 1)
        ) * self.delta

    def add_functionality(self):
        l_markersize = 2
        self.patches = self.left_rectangles_poly()

        for pat in self.patches:
            self.ax.add_patch(pat)

        self.calculated = self.calc_left_rectangles()
        t = plt.figtext(0.01, 0.07, f"Calc: {self.calculated}")
        tr = plt.figtext(0.01, 0.04, f"Act: {self.actual_value}")
        td = plt.figtext(0.01, 0.01, "Diff: 0")

        tdelta = plt.figtext(0.4, 0.1, f"h: {self.delta}")
        tddelta = plt.figtext(0.4, 0.07, f"diff/h: {self.actual_value / self.delta}")
        tdelta2 = plt.figtext(0.4, 0.04, f"h^2: {self.delta ** 2}")
        tddelta2 = plt.figtext(0.4, 0.01, f"diff/h^2: {self.actual_value / self.delta ** 2}")

        pn, = plt.plot(self.nodes_f_plot(), self.calc_f_dots(self.cur_func), 'go', markersize=3)

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

        def ax_get(n):
            sf = 0.22
            h = 0.025
            return sf - h * n

        axcolor = 'lightgoldenrodyellow'
        axint = plt.axes([0.25, ax_get(0), 0.65, 0.03], facecolor=axcolor)
        axdiv = plt.axes([0.25, ax_get(1), 0.65, 0.03], facecolor=axcolor)
        # axdeg = plt.axes([0.25, ax_get(2), 0.65, 0.03], facecolor=axcolor)
        axrange = plt.axes([0.25, ax_get(2), 0.65, 0.03], facecolor=axcolor)
        axcoeff = plt.axes([0.25, ax_get(3), 0.65, 0.03], facecolor=axcolor)
        axnint = plt.axes([0.25, ax_get(4), 0.65, 0.03], facecolor=axcolor)

        sints = Slider(axint, 'Intervals', 1, 10**self.intervals_mult, valinit=self.intervals_count, valstep=1, color='green')
        sintmult = Slider(axdiv, 'Int Divs', 1, 10, valinit=self.intervals_mult, valstep=1, color='lime')
        # sdeg = Slider(axdeg, 'Degree', 1, 10, valinit=self.degree, valstep=1)
        srange = RangeSlider(axrange, 'Range', -20, 20, valinit=(self.begin, self.end))
        scoeff = Slider(axcoeff, 'Coeff', 0, 10, valinit=self.coeff, color='red')
        snint = Slider(axnint, 'Num Int', 0, self.intervals_count - 1, valinit=0, valstep=1, color='cyan')
        snint.ax.set_visible(False)

        def range_update(val=""):
            self.begin, self.end = srange.val
            self.coeff = scoeff.val
            t, ft = self.get_raw_function(self.cur_func)
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
            # self.degree = sdeg.val

            self.intervals_count = sints.val
            self.intervals_mult = sintmult.val
            sints.valmax = 10 ** self.intervals_mult
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
        sintmult.on_changed(update)
        # sdeg.on_changed(update)
        srange.on_changed(range_update)
        scoeff.on_changed(range_update)
        snint.on_changed(update)

        rax = plt.axes([0.025, 0.65, 0.15, 0.3], facecolor=axcolor)
        radio = RadioButtons(rax, ['Left', 'Right', 'Central', 'Trapezoid'], active=0)

        def change_mode(val):
            if check_bools[3]:
                for patch in self.patches:
                    patch.remove()

                match val:
                    case 'Left':
                        self.patches = self.left_rectangles_poly()
                    case 'Right':
                        self.patches = self.right_rectangles_poly()
                    case 'Central':
                        self.patches = self.central_rectangles_poly()
                    case 'Trapezoid':
                        self.patches = self.trapezoid_poly()

                for patch in self.patches:
                    self.ax.add_patch(patch)

            match val:
                case 'Left':
                    self.calculated = self.calc_left_rectangles()
                case 'Right':
                    self.calculated = self.calc_right_rectangles()
                case 'Central':
                    self.calculated = self.calc_central_rectangles()
                case 'Trapezoid':
                    self.calculated = self.calc_trapezoid()

            pn.set_data(self.nodes_f_plot(), self.calc_f_dots(self.cur_func))
            y_mi, y_ma = self.min_max_on_plot()
            y_diff = (y_ma - y_mi) / 100
            ni.set_data(self.nodes_f_plot()[snint.val:snint.val + 2], [y_mi - y_diff] * 2)

            t.set(text=f"Calc: {self.calculated}")
            tr.set(text=f"Act: {self.actual_value}")
            diff = self.calculated - self.actual_value
            td.set(text=f"Diff: {diff}")
            tdelta.set(text=f"h: {self.delta}")
            tdelta2.set(text=f"h^2: {self.delta ** 2}")

            tddelta.set(text=f"diff/h: {diff / self.delta}")
            tddelta2.set(text=f"diff/h^2: {diff / self.delta ** 2}")

            self.fig.canvas.draw_idle()

        radio.on_clicked(change_mode)

        check_labels = ["Toggle function", "Toggle nodes", "Show Interval", "Toggle Rets"]
        check_bools = [True, True, False, True]

        cax = plt.axes([0.025, 0.48, 0.15, 0.15], facecolor=axcolor)
        check = CheckButtons(cax, check_labels, check_bools)

        def turn_checks(val):
            if val == check_labels[0]:
                self.function_plot.set_visible(not self.function_plot.get_visible())
            elif val == check_labels[1]:
                # l.set_markersize(0 if pn.get_visible() else l_markersize)
                pn.set_visible(not pn.get_visible())
            elif val == check_labels[2]:
                ni.set_visible(not ni.get_visible())
                snint.ax.set_visible(ni.get_visible())
            elif val == check_labels[3]:
                check_bools[3] = not check_bools[3]
            self.fig.canvas.draw_idle()

        check.on_clicked(turn_checks)

        y_b = 0.41

        axb = plt.axes([0.025, y_b, 0.1, 0.05], facecolor=axcolor)
        button = Button(axb, "Show table")

        def show_table(val):
            data = [[]] * 4
            data[0] = self.dots
            data[1] = self.calc_f(self.cur_func)
            data[2] = self.patches
            data[3] = [data[1][i] - data[2][i] for i in range(len(self.dots))]

            self.table(('Dot', 'Func', 'Interpolated', 'R = F - L'), data)

        button.on_clicked(show_table)

        axb2 = plt.axes([0.025, y_b - 0.06, 0.1, 0.05], facecolor=axcolor)
        button2 = Button(axb2, "Evaluate")

        def evaluate(val):
            # self.actual_value = integral_req(self.f_str(), *srange.val)
            self.actual_value = quad(f, *srange.val)[0]
            update()

        button2.on_clicked(evaluate)

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

        t, ft = self.get_raw_function(self.cur_func)
        self.function_plot, = plt.plot(t, ft, 'bo', markersize=0, linestyle='-', linewidth=2)

        self.set_coordinates()

        self.add_functionality()


# write here wolfram api appid for full data
wolfram_appid = os.environ.get('WOLFRAM_APPID', "")

f = ()

if __name__ == "__main__":
    p = Plotter()
    f = p.f
    p.start()
