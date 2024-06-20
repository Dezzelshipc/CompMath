from manim import *
from manim_slides import Slide, ThreeDSlide
import numpy as np

# QUALITIES
config.quality = "example_quality"
# config.quality = "high_quality"
# config.quality = "production_quality"

class Intro(Slide):
    def construct(self):
        self.add(Tex("авы",
        #  tex_template=TexTemplateLibrary.default
         )
         )
        t1 = Tex("1").to_corner(DL).set_opacity(0.1)
        self.add(t1)
         
        self.wait()
        self.next_slide()
        self.remove(t1)

        logo = SVGMobject("assets/logo").scale(1.5)
        self.play(Write(logo), run_time=2)
        self.wait(0.1)

        self.play(LaggedStart( ApplyWave(logo), Circumscribe(logo, Circle), lag_ratio=0.25))

        names = Text("Держапольский Юрий Витальевич\nМакарова Виктория Александовна\nБ9121-01.03.02сп").scale(0.5).to_corner(DL)

        title = VGroup(
            Text("Модели конкуренции в экологии и экономике").scale(1.2),
            Text("Взаимодействие трёх популяций").move_to(DOWN)
        ).scale(0.7).move_to(ORIGIN)

        self.play(
            logo.animate.scale(0.5).to_corner(DR),
            Write(names),
            Write(title)
        )

        self.next_slide()

        self.play(
            FadeOut(logo, shift=DOWN),
            FadeOut(title, shift=DOWN),
            FadeOut(names, shift=DOWN)
        )

        self.wait()


class ModelLV(Slide):
    def construct(self):

        sgroup = VGroup(
            box_s := Rectangle(height=1.5, width=4).shift(UP *1.2),
            Text(r"Жертва").move_to(box_s),
            box_h1 := Rectangle(height=1.5, width=4).shift(LEFT*3 + DOWN *1.2),
            Text(r"Хищник 1").move_to(box_h1),
            box_h2 := Rectangle(height=1.5, width=4).shift(RIGHT*3 + DOWN*1.2 ),
            Text(r"Хищник 2").move_to(box_h2),
        )

        self.play(Write(sgroup))
        self.next_slide()

        arr = Arrow(buff=0, start=box_s.get_edge_center(DOWN)+LEFT, end=box_h1.get_edge_center(UP)+RIGHT)
        arr2 = Arrow(buff=0, start=box_s.get_edge_center(DOWN)+RIGHT, end=box_h2.get_edge_center(UP)+LEFT)
        arr3 = Arrow(buff=0, start=box_h1.get_edge_center(RIGHT), end=box_h2.get_edge_center(LEFT))
        arr4 = Arrow(buff=0, start=box_h2.get_edge_center(RIGHT), end=box_h2.get_edge_center(RIGHT)+RIGHT)

        sgroup += arr
        sgroup += arr2
        sgroup += arr3
        sgroup += arr4

        self.play(GrowArrow(arr), GrowArrow(arr2), GrowArrow(arr3), GrowArrow(arr4))
        self.next_slide()

        xgroup = VGroup(
            MathTex("x_1").next_to(box_s,DOWN),
            MathTex("x_2").next_to(box_h1,DOWN),
            MathTex("x_3").next_to(box_h2,DOWN),
        )
        sgroup += xgroup
        self.play(Write(xgroup))
        self.next_slide()
        
        tex = MathTex(r"\dot{x} = f(x)").shift(DOWN*2)
        tt = MathTex("x = (x_1, x_2, x_3)").next_to(tex, DOWN)

        self.play(sgroup.animate.shift(UP))
        self.play(Write(tex), Write(tt))
        self.next_slide()


        self.play(FadeOut(sgroup), FadeOut(tex), FadeOut(tt))
        title = Text("Модель Лотки-Вольтерры")
        self.play(Write(title))
        self.next_slide()

        self.play(title.animate.to_edge(UP))

        lv2 = MathTex(r"""
        \begin{cases}
            \dot{x} = \alpha(x) - V(x)y \\
            \dot{y} = k V(x)y -\beta(y)
        \end{cases}
        """)

        self.play(Write(lv2))
        self.next_slide()

        lv2l = MathTex(r"""
        \begin{cases}
            \dot{x} = \alpha x - v x y \\
            \dot{y} = k v x y - \beta y
        \end{cases}
        """)

        self.play(TransformMatchingShapes(lv2, lv2l))
        self.next_slide()

        sgroup.scale(0.4).to_edge(DL)
        self.play(FadeIn(sgroup))
        self.next_slide()


        self.play(lv2l.animate.to_edge(DR))
        lv2.to_edge(DR)

        lv3f = MathTex(r"""
        \begin{cases}
            \dot{x}_1 = f_1(x_1, x_2, x_3) \\
            \dot{x}_2 = f_2(x_1, x_2, x_3) \\
            \dot{x}_3 = f_3(x_1, x_2, x_3) \\
        \end{cases}
        """)
        self.play(Write(lv3f), TransformMatchingShapes(lv2l, lv2))
        self.next_slide()


        lv3 = MathTex(r"""
        \begin{cases}
            \dot{x}_1 = \varepsilon_1(x_1) - V_{12}(x_1)x_2 - V_{13}(x_1)x_3 \\
            \dot{x}_2 = \varepsilon_2(x_2) + k_{12} V_{12}(x_1)x_2 - V_{23}(x_2)x_3 \\
            \dot{x}_3 = -\varepsilon_3(x_3) + k_{13} V_{13}(x_1)x_3 + k_{23} V_{23}(x_2)x_3
        \end{cases}
        """)
        self.play(TransformMatchingShapes(lv3f, lv3))
        self.next_slide()

        lv3l = MathTex(r"""
        \begin{cases}
            \dot{x}_1 = \varepsilon_1 x_1 - \alpha_{12} x_1 x_2 - \alpha_{13} x_1 x_3 \\
            \dot{x}_2 = \varepsilon_2 x_2 + k_{12} \alpha_{12} x_1 x_2 - \alpha_{23} x_2 x_3 \\
            \dot{x}_3 = -\varepsilon_3 x_3 + k_{13} \alpha_{13} x_1 x_3 + k_{23} \alpha_{23} x_2 x_3
        \end{cases}
        """)

        self.play(TransformMatchingShapes(lv3, lv3l), TransformMatchingShapes(lv2, lv2l))
        self.next_slide()

        self.play(FadeOut(lv3l), FadeOut(sgroup), FadeOut(lv2l))
        
        an_t = Text("Анализ").next_to(title, DOWN)

        self.play(Write(an_t))
        

        balance = MathTex(r"\dot{x} = 0 ~ \Rightarrow ~ f(x^*) = 0").move_to( UP)
        firstlin = MathTex(r"J = \frac{\partial f}{\partial x} \qquad A = J\big|_{x^*}")
        detlin = MathTex(r"\det(\lambda I - A) = 0 \qquad \forall i: \Re (\lambda_i) < 0").move_to(DOWN)

        self.play(Write(balance), run_time=1)
        self.play(Write(firstlin), run_time=1)
        self.play(Write(detlin), run_time=1)
        self.next_slide()

        self.play(
            Unwrite(title, run_time=1),
            Unwrite(an_t),
            FadeOut(balance, shift=DOWN),
            FadeOut(firstlin, shift=DOWN),
            FadeOut(detlin, shift=DOWN),
        )

        self.wait(3)
        self.next_slide()


class LV2D1(Slide, MovingCameraScene):
    def construct(self):
        
        lv3 = MathTex(r"""\begin{cases}
            \dot{x}_1 = 10 x_1 - 6 x_1 x_2 - 2 x_1 x_3 \\
            \dot{x}_2 = 8 x_2 + 4 \cdot 6 x_1 x_2 - 0.5 x_2 x_3 \\
            \dot{x}_3 = -6 x_3 + 1 \cdot 2 x_1 x_3 + 0.5 \cdot 0.5 x_2 x_3
        \end{cases}""")

        model_name = Text("Модель 1").to_edge(UP)

        self.play(Write(lv3), Write(model_name))
        self.next_slide()
        self.play(Unwrite(lv3), Unwrite(model_name), run_time=1)


        ksi1, ksi2, ksi3 = 10, 8, 6
        a12, a13, a23 = 6, 2, 0.5
        k12, k13, k23 = 4, 1, 0.5
        def right_x1(x):
            x= x*10
            return np.array([
                (-ksi2 - a23 * x[1]) * x[0],
                (-ksi3  + k23 * a23 * x[0]) * x[1],
                0
            ])/50

        def right_x2(x):
            x= x*2
            return np.array([
                (ksi1 - a13 * x[1]) * x[0],
                (-ksi3 + k13 * a13 * x[0]) * x[1],
                0
            ])/10

        def right_x3(x):
            x = x*10
            return np.array([
                (ksi1 - a12 * x[1]) * x[0],
                (-ksi2 + k12 * a12 * x[0]) * x[1],
                0
            ])/100
        

        self.camera.frame.save_state()

        vector_field1 = ArrowVectorField(
            right_x1,
            x_range=[0,8,0.5],
            y_range=[0,6,0.5],
        )

        axes = Axes(
            x_range=[0,60,10],
            y_range=[0,60,10],
            x_length=6,
            y_length=6,
        ).add_coordinates()

        ax_labels = axes.get_axis_labels(
            x1l := MathTex("x_2"), x2l := MathTex("x_3")
        )

        SHIFT = -axes.c2p(0,0,0)

        axes.shift(SHIFT)
        ax_labels.shift(SHIFT)
        ax_labels[0].shift(DOWN*0.7)
        ax_labels[1].shift(LEFT*0.7)

        self.camera.frame.move_to(axes.c2p(30,30,0))

        x10t = MathTex("x_1 = 0").to_corner(UL).shift(SHIFT)
        
        self.play(Write(axes), Write(ax_labels), Write(x10t))
        self.play(Write(vector_field1))
        self.play(Indicate(x10t))
        self.wait(1)
        self.next_slide()

        self.play(vector_field1.animate.set_opacity(0.5))


        dots_group = VGroup()
        dots_group += Dot(color=GREEN).move_to(axes.c2p(5,5,0))
        dots_group += Dot(color=BLUE).move_to(axes.c2p(10,10,0))
        dots_group += Dot(color=YELLOW).move_to(axes.c2p(15,15,0))
        dots_group += Dot(color=PINK).move_to(axes.c2p(20,20,0))

        dots_group += Dot(color=WHITE).move_to(axes.c2p(25,15,0))

        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(0.1,0,0))
        dots_group += Dot(color=PURE_BLUE).move_to(axes.c2p(0,10,0))


        self.play(Create(dots_group))
        self.next_slide()

        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field1.nudge(dot, 0, 60)
            dot.add_updater(vector_field1.get_nudge_updater())

            self.add(path)
            grp+=path

        dots_group += grp
        self.wait(6)

        self.next_slide()
        for dot in dots_group:
            dot.clear_updaters()

        self.play(ShrinkToCenter(dots_group))

        

        vector_field2 = ArrowVectorField(
            right_x2,
            x_range=[0,8,0.5],
            y_range=[0,6,0.5],
        ).set_opacity(0.5)
        axes2 = Axes(
            x_range=[0,12,2],
            y_range=[0,12,2],
            x_length=6,
            y_length=6,
        ).add_coordinates().shift(SHIFT)


        x20t = MathTex("x_2 = 0").move_to(x10t)
        
        self.play(
            TransformMatchingShapes(x10t, x20t), 
            ReplacementTransform(vector_field1, vector_field2), 
            ReplacementTransform(axes, axes2),
            Transform(x1l, MathTex("x_1").move_to(x1l) ))
        self.play(Indicate(x20t))

        self.next_slide()

        dots_group = VGroup()
        dots_group += Dot(color=GREEN).move_to(axes2.c2p(2,1,0))
        dots_group += Dot(color=BLUE).move_to(axes2.c2p(2,1.5,0))
        dots_group += Dot(color=YELLOW).move_to(axes2.c2p(2,2,2,0))
        dots_group += Dot(color=PINK).move_to(axes2.c2p(2,2.5,0))

        dots_group += Dot(color=WHITE).move_to(axes2.c2p(2.5,5,0))

        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(0.1,0,0))
        dots_group += Dot(color=PURE_BLUE).move_to(axes.c2p(0,10,0))

        self.play(Create(dots_group))
        self.next_slide()

        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field2.nudge(dot, 0, 60)
            dot.add_updater(vector_field2.get_nudge_updater())

            self.add(path)
            grp+=path

        dots_group += grp
        self.wait(6)
        self.next_slide()


        for dot in dots_group:
            dot.clear_updaters()

        self.play(ShrinkToCenter(dots_group))

        

        vector_field3 = ArrowVectorField(
            right_x3,
            x_range=[0,8,0.5],
            y_range=[0,6,0.5],
        ).set_opacity(0.5)
        axes3 = Axes(
            x_range=[0,60,10],
            y_range=[0,60,10],
            x_length=6,
            y_length=6,
        ).add_coordinates().shift(SHIFT)


        x30t = MathTex("x_3 = 0").move_to(x10t)
        

        self.play(
            TransformMatchingShapes(x20t, x30t), 
            ReplacementTransform(vector_field2, vector_field3), 
            ReplacementTransform(axes2, axes3),
            Transform(x2l, MathTex("x_2").move_to(x2l)))
        self.play(Indicate(x30t))

        self.next_slide()

        dots_group = VGroup()
        dots_group += Dot(color=GREEN).move_to(axes.c2p(5,5,0))
        dots_group += Dot(color=BLUE).move_to(axes.c2p(10,5,0))
        dots_group += Dot(color=YELLOW).move_to(axes.c2p(15,5,0))
        dots_group += Dot(color=PINK).move_to(axes.c2p(20,5,0))

        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(0.1,0,0))
        dots_group += Dot(color=PURE_BLUE).move_to(axes.c2p(0,10,0))

        self.play(Create(dots_group))
        self.next_slide()

        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field3.nudge(dot, 0, 120)
            dot.add_updater(vector_field3.get_nudge_updater())

            self.add(path)
            grp+=path

        dots_group += grp
        self.wait(1)
        self.next_slide()
        for dot in dots_group:
            dot.clear_updaters()

        
        self.play(
            FadeOut(x30t, shift=DOWN), 
            FadeOut(vector_field3, scale=0.5), 
            FadeOut(axes3, scale=0.5), 
            FadeOut(ax_labels, scale=0.5), 
            ShrinkToCenter(dots_group)
        )

        self.next_slide()


class LV3D1(ThreeDSlide):
    def construct(self):

        ksi1, ksi2, ksi3 = 10, 8, 6
        a12, a13, a23 = 6, 2, 0.5
        k12, k13, k23 = 4, 1, 0.5
        def right(x):
            x=x + 3
            x = x * 10
            return np.array([
                (ksi1 - a12 * x[1] - a13 * x[2]) * x[0],
                (ksi2 + k12 * a12 * x[0] - a23 * x[2]) * x[1],
                (-ksi3 + k13 * a13 * x[0] + k23 * a23 * x[1]) * x[2]
            ])/70

        axes = ThreeDAxes(
            x_range=[0,60,10],
            y_range=[0,60,10],
            z_range=[0,60,10],
            x_length=6,
            y_length=6,
            z_length=6
        ).shift(IN * 3).add_coordinates()

        ax_labels = axes.get_axis_labels(
            MathTex("x_1"), MathTex("x_2"), MathTex("x_3")
        )

        vector_field = ArrowVectorField(
            right,
            color=BLUE,
            three_dimensions=True,
            x_range=[-3,3,1],
            y_range=[-3,3,1],
            z_range=[-3,3,1]
        )

        self.set_camera_orientation(phi=75*DEGREES, theta=30*DEGREES)
        phi, theta, focal_distance, gamma, distance_to_origin = self.camera.get_value_trackers()

        self.play(Write(axes), Write(ax_labels), Write(vector_field))

        self.next_slide()

        self.play(vector_field.animate.set_opacity(0.2))


        dots_group = VGroup()
        dots_group += Dot3D(color=GREEN)    .move_to(axes.c2p(5,5,5))
        dots_group += Dot3D(color=BLUE)     .move_to(axes.c2p(10,10,10))
        dots_group += Dot3D(color=YELLOW)   .move_to(axes.c2p(15,15,20))
        dots_group += Dot3D(color=PINK)     .move_to(axes.c2p(5,9,11))

        dots_group += Dot3D(color=ORANGE)    .move_to(axes.c2p(0,60,20))

        dots_group += Dot3D(color=WHITE)    .move_to(axes.c2p(10,0,10))
        dots_group += Dot3D(color=GREY)     .move_to(axes.c2p(5,0,5))

        dots_group += Dot3D(color=PURPLE)   .move_to(axes.c2p(5,1,1))

        self.play(Create(dots_group))
        self.next_slide()


        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field.nudge(dot, 0, 120)
            dot.add_updater(vector_field.get_nudge_updater())

            self.add(path)
            grp+=path

        for p in grp:
            dots_group += p

        self.wait(6)

        self.next_slide()


        self.play(theta.animate.set_value((360+30) * DEGREES), run_time=8, rate_func=rate_functions.smootherstep)
        # self.wait()

        self.next_slide()

        for d in dots_group:
            d.clear_updaters()

        self.play(ShrinkToCenter(dots_group), Unwrite(axes), ShrinkToCenter(vector_field), Unwrite(ax_labels))
        self.wait()


        self.next_slide()
        

class ModelK(Slide):
    def construct(self):
        title = Text("Модель Колмогорова")

        self.play(Write(title))
        self.next_slide()

        self.play(title.animate.to_edge(UP))

        k2g = MathTex(r"""\begin{cases}
            \dot{x} = \alpha(x)x - V(x) y \\
            \dot{y} = K(x) y
        \end{cases}""")

        k2 = k2g.copy()

        self.play(Write(k2))
        self.next_slide()

        assump = MathTex(r"""\begin{split}
            \alpha' < 0 \hspace{2em} \alpha(0) > 0 > \alpha(\infty) \\
            K' > 0 \hspace{2em} K(0) < 0 < K(\infty) \\
            V(x) > 0, ~ x > 0 \hspace{2em} V(0) = 0 \\
        \end{split}""").to_edge(DOWN)

        self.play(Write(assump))
        self.next_slide()

        sgroup = VGroup(
            box_s := Rectangle(height=1.5, width=4).shift(UP *1.2),
            Text(r"Жертва").move_to(box_s),
            box_h1 := Rectangle(height=1.5, width=4).shift(LEFT*3 + DOWN *1.2),
            Text(r"Хищник 1").move_to(box_h1),
            box_h2 := Rectangle(height=1.5, width=4).shift(RIGHT*3 + DOWN*1.2 ),
            Text(r"Хищник 2").move_to(box_h2),

            Arrow(buff=0, start=box_s.get_edge_center(DOWN)+LEFT, end=box_h1.get_edge_center(UP)+RIGHT),
            Arrow(buff=0, start=box_s.get_edge_center(DOWN)+RIGHT, end=box_h2.get_edge_center(UP)+LEFT),
            Arrow(buff=0, start=box_h1.get_edge_center(RIGHT), end=box_h2.get_edge_center(LEFT)),
            Arrow(buff=0, start=box_h2.get_edge_center(RIGHT), end=box_h2.get_edge_center(RIGHT)+RIGHT),

            MathTex("x_1").next_to(box_s,DOWN),
            MathTex("x_2").next_to(box_h1,DOWN),
            MathTex("x_3").next_to(box_h2,DOWN),
        ).scale(0.4).to_edge(DL)

        k3f = MathTex(r"""
        \begin{cases}
            \dot{x}_1 = f_1(x_1, x_2, x_3) \\
            \dot{x}_2 = f_2(x_1, x_2, x_3) \\
            \dot{x}_3 = f_3(x_1, x_2, x_3) \\
        \end{cases}
        """)

        k3 = k3f.copy()

        self.play(FadeIn(sgroup), FadeOut(assump, shift=DOWN), k2.animate.to_edge(DR))
        self.play(Write(k3))
        self.wait()
        self.next_slide()

        k3g = MathTex(r"""\begin{cases}
            \dot{x}_1 = \varepsilon(x_1)x_1 - V_{12}(x_1)x_2 - V_{13}(x_1)x_3 \\
            \dot{x}_2 = K_{12}(x_1)x_2 - V_{23}(x_2)x_3 \\
            \dot{x}_3 = K_{13}(x_1)x_3 + K_{23}(x_2)x_3 \\ 
        \end{cases}""")

        self.play(TransformMatchingShapes(k3, k3g))
        self.next_slide()

        assump = MathTex(r"""\begin{split}
            \varepsilon' < 0 \hspace{1em} \varepsilon(0) > 0 > \varepsilon(\infty) \\
            K_{ij}' > 0 \hspace{1em} K_{ij}(0) < 0 < K_{ij}(\infty) \\
            V_{ij}(x_i) > 0, ~ x_i > 0 \hspace{1em} V_{ij}(0) = 0 \\
        \end{split}""").to_corner(DR)

        self.play(LaggedStart(FadeOut(k2, shift=DOWN), FadeIn(assump,shift=DOWN), lag_ratio=0.25))
        self.next_slide()

        k3l = MathTex(r"""\begin{cases}
            \dot{x}_1 = \smash{ \overbrace{ \left( -\varepsilon x_1 + \delta \right) }^{\varepsilon(x_1)}}  x_1 - \smash{ \overbrace{ v_{12} x_1}^{V_{12}(x_1)}} x_2 - v_{13} x_1 x_3 \\
            \dot{x}_2 = \left( k_{12} x_1 - m_{12} \right)x_2 - v_{23} x_2 x_3 \\
            \dot{x}_3 = \smash{ \underbrace{ \left( k_{13} x_1 - m_{13} \right) }_{K_{13}(x_1)}} x_3 + \left( k_{23} x_2 - m_{23} \right)x_3 \\
        \end{cases}""")

        self.play(TransformMatchingShapes(k3g, k3l))
        self.next_slide()

        self.play(Unwrite(title), Unwrite(assump), Unwrite(sgroup, run_time=2), Unwrite(k3l))
        self.wait()

        self.next_slide()

        
class K2D1(Slide, MovingCameraScene):
    def construct(self):

        def right_x1(x):
            x= x*2
            return np.array([
                (( - 5) * x[0] - 1 * x[1] * x[0]),
                ((- 3) * x[1] + (1 * x[0] - 4) * x[1]),
                0
            ])/10

        def right_x2(x):
            x= x*2
            return np.array([
                ((-1 * x[0] + 10) * x[0] - 3 * x[1] * x[0]),
                ((1 * x[0] - 3) * x[1] + (- 4) * x[1]),
                0
            ])/5

        def right_x3(x):
            x = x*2
            return np.array([
                ((-1 * x[0] + 10) * x[0] - 2 * x[1] * x[0]),
                ((1 * x[0] - 5) * x[1]),
                0
            ])/5
        

        self.camera.frame.save_state()

        vector_field1 = ArrowVectorField(
            right_x1,
            # padding=0.01,
            color=BLUE,
            x_range=[0,6,0.5],
            y_range=[0,6,0.5],
        )

        # Axes._origin_shift = lambda *x: 0
        axes = Axes(
            x_range=[0,12,2],
            y_range=[0,12,2],
            x_length=6,
            y_length=6,
        ).add_coordinates()

        ax_labels = axes.get_axis_labels(
            x1l := MathTex("x_2"), x2l := MathTex("x_3")
        )

        SHIFT = -axes.c2p(0,0,0)

        axes.shift(SHIFT)
        ax_labels.shift(SHIFT)
        ax_labels[0].shift(DOWN*0.7)
        ax_labels[1].shift(LEFT*0.7)

        self.camera.frame.move_to(axes.c2p(6,6,0))

        x10t = MathTex("x_1 = 0").to_corner(UL).shift(SHIFT)
        
        self.play(Write(axes), Write(ax_labels), Write(x10t))
        self.play(Write(vector_field1))
        self.play(Indicate(x10t))
        self.wait(1)
        self.next_slide()

        self.play(vector_field1.animate.set_opacity(0.5))


        dots_group = VGroup()
        dots_group += Dot(color=GREEN).move_to(axes.c2p(12,2,0))
        dots_group += Dot(color=BLUE).move_to(axes.c2p(15,5,0))
        dots_group += Dot(color=YELLOW).move_to(axes.c2p(17,7,0))
        dots_group += Dot(color=PINK).move_to(axes.c2p(12,10,0))

        dots_group += Dot(color=WHITE).move_to(axes.c2p(15,15,0))

        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(10,0,0))
        dots_group += Dot(color=PURE_BLUE).move_to(axes.c2p(0,10,0))

        self.play(Create(dots_group))
        self.next_slide()

        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field1.nudge(dot, 0, 60)
            dot.add_updater(vector_field1.get_nudge_updater())

            self.add(path)
            grp+=path

        dots_group += grp
        self.wait(6)

        self.next_slide()
        for dot in dots_group:
            dot.clear_updaters()

        self.play(ShrinkToCenter(dots_group))

        

        vector_field2 = ArrowVectorField(
            right_x2,
            # padding=0.01,
            color=BLUE,
            x_range=[0,6,0.5],
            y_range=[0,6,0.5],
        ).set_opacity(0.5)


        x20t = MathTex("x_2 = 0").move_to(x10t)
        
        self.play(
            TransformMatchingShapes(x10t, x20t), 
            ReplacementTransform(vector_field1, vector_field2),
            Transform(x1l, MathTex("x_1").move_to(x1l) )
        )
        self.play(Indicate(x20t))

        self.next_slide()

        dots_group = VGroup()
        dots_group += Dot(color=GREEN).move_to(axes.c2p(2,2,0))
        dots_group += Dot(color=BLUE).move_to(axes.c2p(10,5,0))
        dots_group += Dot(color=YELLOW).move_to(axes.c2p(7,7,0))
        dots_group += Dot(color=PINK).move_to(axes.c2p(10,10,0))

        dots_group += Dot(color=PURPLE).move_to(axes.c2p(12,3,0))
        dots_group += Dot(color=WHITE).move_to(axes.c2p(15,15,0))
        dots_group += Dot(color=ORANGE).move_to(axes.c2p(15,1,0))

        dots_group += Dot(color=PURE_RED).move_to(axes.c2p(10,0.1,0))
        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(0.1,0,0))
        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(20,0,0))
        dots_group += Dot(color=PURE_BLUE).move_to(axes.c2p(0,10,0))

        self.play(Create(dots_group))
        self.next_slide()

        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field2.nudge(dot, 0, 60)
            dot.add_updater(vector_field2.get_nudge_updater())

            self.add(path)
            grp+=path

        dots_group += grp
        self.wait(6)
        self.next_slide()


        for dot in dots_group:
            dot.clear_updaters()

        self.play(ShrinkToCenter(dots_group))

        

        vector_field3 = ArrowVectorField(
            right_x3,
            color=BLUE,
            x_range=[0,6,0.5],
            y_range=[0,6,0.5],
        ).set_opacity(0.5)


        x30t = MathTex("x_3 = 0").move_to(x10t)
        

        self.play(
            TransformMatchingShapes(x20t, x30t), 
            Transform(vector_field2, vector_field3),
            Transform(x2l, MathTex("x_2").move_to(x2l))
            )
        self.play(Indicate(x30t))

        self.next_slide()

        dots_group = VGroup()
        dots_group += Dot(color=GREEN).move_to(axes.c2p(2,2,0))
        dots_group += Dot(color=BLUE).move_to(axes.c2p(10,5,0))
        dots_group += Dot(color=YELLOW).move_to(axes.c2p(7,7,0))
        dots_group += Dot(color=PINK).move_to(axes.c2p(10,10,0))

        dots_group += Dot(color=PURPLE).move_to(axes.c2p(12,3,0))
        dots_group += Dot(color=WHITE).move_to(axes.c2p(15,15,0))
        dots_group += Dot(color=ORANGE).move_to(axes.c2p(15,1,0))

        dots_group += Dot(color=PURE_RED).move_to(axes.c2p(10,0.1,0))
        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(0.1,0,0))
        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(20,0,0))
        dots_group += Dot(color=PURE_BLUE).move_to(axes.c2p(0,10,0))

        self.play(Create(dots_group))
        self.next_slide()

        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field3.nudge(dot, 0, 120)
            dot.add_updater(vector_field3.get_nudge_updater())

            self.add(path)
            grp+=path

        dots_group += grp
        self.wait(6)
        self.next_slide()
        for dot in dots_group:
            dot.clear_updaters()
        
        self.play(
            FadeOut(x30t, shift=DOWN), 
            FadeOut(vector_field3, scale=0.5), 
            FadeOut(axes, scale=0.5), 
            FadeOut(ax_labels, scale=0.5), 
            ShrinkToCenter(dots_group))

        self.next_slide()


class K3D1(ThreeDSlide):
    def construct(self):

        def right(x):
            x = x + 2
            x = x * 2
            return np.array([
                ((-1 * x[0] + 10) * x[0] - 2 * x[1] * x[0] - 3 * x[2] * x[0]),
                ((1 * x[0] - 5) * x[1] - 1 * x[2] * x[1]),
                ((1 * x[0] - 3) * x[2] + (1 * x[1] - 4) * x[2])
            ])/10

        def right_p(x):
            x = x + np.array([11/2, 3/2, 1/2])/2
            x = x * 2
            return np.array([
                ((-1 * x[0] + 10) * x[0] - 2 * x[1] * x[0] - 3 * x[2] * x[0]),
                ((1 * x[0] - 5) * x[1] - 1 * x[2] * x[1]),
                ((1 * x[0] - 3) * x[2] + (1 * x[1] - 4) * x[2])
            ])/10

        axes = ThreeDAxes(
            x_range=[0,12,2],
            y_range=[0,12,2],
            z_range=[0,10,2],
            x_length=6,
            y_length=6,
            z_length=5
        ).shift(IN * 2+ RIGHT + UP).add_coordinates()

        ax_labels = axes.get_axis_labels(
            MathTex("x_1"), MathTex("x_2"), MathTex("x_3")
        )

        vector_field = ArrowVectorField(
            right,
            color=BLUE,
            three_dimensions=True,
            x_range=[-2,4,1],
            y_range=[-2,4,1],
            z_range=[-2,3,1]
        )

        self.set_camera_orientation(phi=75*DEGREES, theta=-40*DEGREES)

        self.play(Write(axes), Write(ax_labels), Write(vector_field))
        self.wait()

        self.next_slide()

        self.play(vector_field.animate.set_opacity(0.2))


        dots_group = VGroup()
        dots_group += Dot3D(color=PURE_RED).move_to(axes.c2p(10,0.1,0.1))

        dots_group += Dot3D(color=RED).move_to(axes.c2p(1,1,1))
        dots_group += Dot3D(color=GREEN).move_to(axes.c2p(10,1,1))
        dots_group += Dot3D(color=BLUE).move_to(axes.c2p(1,10,1))
        dots_group += Dot3D(color=YELLOW).move_to(axes.c2p(1,1,10))
        dots_group += Dot3D(color=PURPLE).move_to(axes.c2p(10,10,1))
        dots_group += Dot3D(color=PINK).move_to(axes.c2p(1,10,10))
        dots_group += Dot3D(color=GREY).move_to(axes.c2p(10,1,10))
        dots_group += Dot3D(color=ORANGE).move_to(axes.c2p(10,10,10))

        self.play(Create(dots_group))
        self.next_slide()


        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field.nudge(dot, 0, 120)
            dot.add_updater(vector_field.get_nudge_updater())

            self.add(path)
            grp+=path

        for p in grp:
            dots_group += p

        phi, theta, focal_distance, gamma, distance_to_origin = self.camera.get_value_trackers()
        self.play(theta.animate.set_value((360-40) * DEGREES), run_time=10, rate_func=rate_functions.smootherstep)
        self.wait(0.1)

        self.next_slide()

        for d in dots_group:
            d.clear_updaters()

        self.play(ShrinkToCenter(dots_group))
        
        self.next_slide()

        vector_field2 = ArrowVectorField(
            right_p,
            three_dimensions=True,
            x_range=[-1,1,0.4],
            y_range=[-3/4,1,0.4],
            z_range=[-1/4,1,0.4]
        ).set_opacity(0.2)


        self.set_camera_orientation(phi=75*DEGREES, theta=-40*DEGREES)


        SHIFT = -axes.c2p(11/2,3/2,1/2)
        self.play(
            axes.animate.shift(SHIFT), 
            ax_labels.animate.shift(SHIFT), 
            ReplacementTransform(vector_field, vector_field2), 
            distance_to_origin.animate.set_value(8),
            focal_distance.animate.set_value(8)
            )
        self.wait()

        udot = Dot3D(axes.c2p(11/2,3/2,1/2), radius=0.005)
        self.add(udot)

        self.play(Flash(udot, line_length=0.04, run_time=1, flash_radius=0.03))
        self.next_slide()
        

        x0s = [
            ([11/2, 3/2, 1], RED),
            ([11/2, 3/2, 1/4], GREEN),
            ([11/2, 1, 1/2], BLUE),
            ([11/2, 2, 1/2], YELLOW),
            ([5, 3/2, 1/2], PINK),
            ([6, 3/2, 1/2], PURPLE),
            ([5, 1, 1/4], GREY),
            ([6, 2, 1], ORANGE),
        ]

        dots_group = VGroup()
        for xi in x0s:
            dots_group += Dot3D(point=axes.c2p(*xi[0]), radius=0.005, color=xi[1])

        self.play(Create(dots_group))

        self.next_slide()


        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field2.nudge(dot, 0, 120)
            dot.add_updater(vector_field2.get_nudge_updater())

            self.add(path)
            grp+=path

        for p in grp:
            dots_group += p

        self.play(
            vector_field2.animate.set_opacity(0),
            run_time=1
        )


        phi, theta, focal_distance, gamma, distance_to_origin = self.camera.get_value_trackers()
        self.play(
            theta.animate.set_value((20) * DEGREES), 
            distance_to_origin.animate.set_value(12),
            focal_distance.animate.set_value(12),
            run_time=8, 
            rate_func=rate_functions.smootherstep
            )
        self.wait(10)

        self.next_slide()

        for d in dots_group:
            d.clear_updaters()

        self.play( 
            Unwrite(axes), 
            Unwrite(ax_labels),
            ShrinkToCenter(dots_group), 
            Uncreate(udot),
            run_time = 2
        )
        self.wait()
        dots_group.set_opacity(0)

        self.next_slide()


class K2D2(Slide, MovingCameraScene):
    def construct(self):

        def right_x1(x):
            x= x*2
            return np.array([
                ((-3 * x[0] + 9) * x[0] + ( - 5) * x[0] - 1 * x[1] * x[0]),
                (( - 3) * x[1] + (1 * x[0] - 4) * x[1]),
                0
            ])/10

        def right_x2(x):
            x= x*2
            return np.array([
                ((-1 * x[0] + 10) * x[0]  - 3 * x[1] * x[0]),
                ((1 * x[0] - 3) * x[1] + ( - 4) * x[1]),
                0
            ])/10

        def right_x3(x):
            x = x*2
            return np.array([
                ((-1 * x[0] + 10) * x[0] - 2 * x[1] * x[0] ),
                ((-3 * x[1] + 9) * x[1] + (1 * x[0] - 5) * x[1] ),
                0
            ])/5
        

        self.camera.frame.save_state()

        vector_field1 = ArrowVectorField(
            right_x1,
            # color=BLUE,
            x_range=[0,6,0.5],
            y_range=[0,6,0.5],
        )

        # Axes._origin_shift = lambda *x: 0
        axes = Axes(
            x_range=[0,12,2],
            y_range=[0,12,2],
            x_length=6,
            y_length=6,
        ).add_coordinates()

        ax_labels = axes.get_axis_labels(
            x1l := MathTex("x_2"), x2l := MathTex("x_3")
        )

        SHIFT = -axes.c2p(0,0,0)

        axes.shift(SHIFT)
        ax_labels.shift(SHIFT)
        ax_labels[0].shift(DOWN*0.7)
        ax_labels[1].shift(LEFT*0.7)

        self.camera.frame.move_to(axes.c2p(6,6,0))

        x10t = MathTex("x_1 = 0").to_corner(UL).shift(SHIFT)
        
        self.play(Write(axes), Write(ax_labels), Write(x10t))
        self.play(Write(vector_field1))
        self.play(Indicate(x10t))
        self.wait(1)
        self.next_slide()

        self.play(vector_field1.animate.set_opacity(0.5))


        dots_group = VGroup()
        dots_group += Dot(color=GREEN).move_to(axes.c2p(12,2,0))
        dots_group += Dot(color=BLUE).move_to(axes.c2p(15,5,0))
        dots_group += Dot(color=YELLOW).move_to(axes.c2p(17,7,0))
        dots_group += Dot(color=PINK).move_to(axes.c2p(12,10,0))
        dots_group += Dot(color=ORANGE).move_to(axes.c2p(1/2,12,0))

        dots_group += Dot(color=WHITE).move_to(axes.c2p(15,15,0))

        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(10,0,0))
        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(0.1,0,0))
        dots_group += Dot(color=PURE_BLUE).move_to(axes.c2p(0,10,0))

        self.play(Create(dots_group))
        self.next_slide()

        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field1.nudge(dot, 0, 60)
            dot.add_updater(vector_field1.get_nudge_updater())

            self.add(path)
            grp+=path

        dots_group += grp
        self.wait(6)

        self.next_slide()
        for dot in dots_group:
            dot.clear_updaters()

        self.play(ShrinkToCenter(dots_group))

        

        vector_field2 = ArrowVectorField(
            right_x2,
            # color=BLUE,
            x_range=[0,6,0.5],
            y_range=[0,6,0.5],
        ).set_opacity(0.5)


        x20t = MathTex("x_2 = 0").move_to(x10t)
        
        self.play(
            TransformMatchingShapes(x10t, x20t), 
            ReplacementTransform(vector_field1, vector_field2),
            Transform(x1l, MathTex("x_1").move_to(x1l) )
        )
        self.play(Indicate(x20t))

        self.next_slide()

        dots_group = VGroup()
        dots_group += Dot(color=GREEN).move_to(axes.c2p(2,2,0))
        dots_group += Dot(color=BLUE).move_to(axes.c2p(10,5,0))
        dots_group += Dot(color=YELLOW).move_to(axes.c2p(7,7,0))
        dots_group += Dot(color=PINK).move_to(axes.c2p(10,10,0))

        dots_group += Dot(color=PURPLE).move_to(axes.c2p(12,3,0))
        dots_group += Dot(color=WHITE).move_to(axes.c2p(2,12,0))
        dots_group += Dot(color=ORANGE).move_to(axes.c2p(15,1,0))

        dots_group += Dot(color=PURE_RED).move_to(axes.c2p(10,0.1,0))
        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(0.1,0,0))
        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(20,0,0))
        dots_group += Dot(color=PURE_BLUE).move_to(axes.c2p(0,10,0))

        self.play(Create(dots_group))
        self.next_slide()

        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field2.nudge(dot, 0, 60)
            dot.add_updater(vector_field2.get_nudge_updater())

            self.add(path)
            grp+=path

        dots_group += grp
        self.wait(8)
        self.next_slide()


        for dot in dots_group:
            dot.clear_updaters()

        self.play(ShrinkToCenter(dots_group))

        

        vector_field3 = ArrowVectorField(
            right_x3,
            # color=BLUE,
            x_range=[0,6,0.5],
            y_range=[0,6,0.5],
        ).set_opacity(0.5)


        x30t = MathTex("x_3 = 0").move_to(x10t)
        

        self.play(
            TransformMatchingShapes(x20t, x30t), 
            Transform(vector_field2, vector_field3),
            Transform(x2l, MathTex("x_2").move_to(x2l))
            )
        self.play(Indicate(x30t))

        self.next_slide()

        dots_group = VGroup()
        dots_group += Dot(color=GREEN).move_to(axes.c2p(5,1/2,0))
        dots_group += Dot(color=BLUE).move_to(axes.c2p(14,5,0))
        dots_group += Dot(color=YELLOW).move_to(axes.c2p(6,10,0))
        dots_group += Dot(color=PINK).move_to(axes.c2p(10,10,0))

        dots_group += Dot(color=PURPLE).move_to(axes.c2p(1/2,1/2,0))
        dots_group += Dot(color=WHITE).move_to(axes.c2p(1,10,0))
        dots_group += Dot(color=ORANGE).move_to(axes.c2p(15,0.1,0))

        dots_group += Dot(color=PURE_RED).move_to(axes.c2p(10,0.1,0))
        dots_group += Dot(color=PURE_RED).move_to(axes.c2p(0.1,4/3,0))
        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(0.1,0,0))
        dots_group += Dot(color=PURE_GREEN).move_to(axes.c2p(20,0,0))
        dots_group += Dot(color=PURE_BLUE).move_to(axes.c2p(0,10,0))
        dots_group += Dot(color=PURE_BLUE).move_to(axes.c2p(0,0.1,0))

        self.play(Create(dots_group))
        self.next_slide()

        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field3.nudge(dot, 0, 120)
            dot.add_updater(vector_field3.get_nudge_updater())

            self.add(path)
            grp+=path

        dots_group += grp
        self.wait(6)
        self.next_slide()
        for dot in dots_group:
            dot.clear_updaters()
        
        vector_field1.set_opacity(0)
        vector_field2.set_opacity(0)

        self.play(
            FadeOut(x30t, shift=DOWN), 
            FadeOut(vector_field3, scale=0.5), 
            FadeOut(axes, scale=0.5), 
            FadeOut(ax_labels, scale=0.5), 
            ShrinkToCenter(dots_group))

        self.next_slide()


class K3D2(ThreeDSlide):
    def construct(self):

        def right_g(x):
            return np.array([
                ((-1 * x[0] + 10) * x[0] - 2 * x[1] * x[0] - 3 * x[2] * x[0]),
                ((-3 * x[1] + 9) * x[1] + (1 * x[0] - 5) * x[1] - 1 * x[2] * x[1]),
                ((1 * x[0] - 3) * x[2] + (1 * x[1] - 4) * x[2])
            ])/10

        def right(x):
            x = x + 2
            x = x * 2
            return right_g(x)

        def right_p(x):
            x = x + np.array([47/11, 30/11, 1/11])/2
            x = x * 2
            return right_g(x)

        axes = ThreeDAxes(
            x_range=[0,12,2],
            y_range=[0,12,2],
            z_range=[0,10,2],
            x_length=6,
            y_length=6,
            z_length=5
        ).shift(IN * 2+ RIGHT + UP).add_coordinates()

        ax_labels = axes.get_axis_labels(
            MathTex("x_1"), MathTex("x_2"), MathTex("x_3")
        )

        vector_field = ArrowVectorField(
            right,
            color=BLUE,
            three_dimensions=True,
            x_range=[-2,4,1],
            y_range=[-2,4,1],
            z_range=[-2,3,1]
        )

        self.set_camera_orientation(phi=75*DEGREES, theta=-40*DEGREES)

        self.play(Write(axes), Write(ax_labels), Write(vector_field))
        self.wait()

        self.next_slide()

        self.play(vector_field.animate.set_opacity(0.2))


        dots_group = VGroup()
        dots_group += Dot3D(color=PURE_RED).move_to(axes.c2p(10,0.1,0.1))

        dots_group += Dot3D(color=RED).move_to(axes.c2p(1,1,1))
        dots_group += Dot3D(color=GREEN).move_to(axes.c2p(10,1,1))
        dots_group += Dot3D(color=BLUE).move_to(axes.c2p(1,10,1))
        dots_group += Dot3D(color=YELLOW).move_to(axes.c2p(1,1,10))
        dots_group += Dot3D(color=PURPLE).move_to(axes.c2p(10,10,1))
        dots_group += Dot3D(color=PINK).move_to(axes.c2p(1,10,10))
        dots_group += Dot3D(color=GREY).move_to(axes.c2p(10,1,10))
        dots_group += Dot3D(color=ORANGE).move_to(axes.c2p(10,10,10))

        self.play(Create(dots_group))
        self.next_slide()


        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field.nudge(dot, 0, 120)
            dot.add_updater(vector_field.get_nudge_updater())

            self.add(path)
            grp+=path

        for p in grp:
            dots_group += p

        phi, theta, focal_distance, gamma, distance_to_origin = self.camera.get_value_trackers()
        self.play(theta.animate.set_value((360-40) * DEGREES), run_time=10, rate_func=rate_functions.smootherstep)
        self.wait(0.1)

        self.next_slide()

        for d in dots_group:
            d.clear_updaters()

        self.play(ShrinkToCenter(dots_group))
        
        self.next_slide()

        vector_field2 = ArrowVectorField(
            right_p,
            three_dimensions=True,
            x_range=[-1,1,0.4],
            y_range=[-3/4,1,0.4],
            z_range=[-1/22,1,0.4]
        ).set_opacity(0.2)


        self.set_camera_orientation(phi=75*DEGREES, theta=-40*DEGREES)


        SHIFT = -axes.c2p(47/11, 30/11, 1/11)
        self.play(
            axes.animate.shift(SHIFT), 
            ax_labels.animate.shift(SHIFT), 
            ReplacementTransform(vector_field, vector_field2), 
            distance_to_origin.animate.set_value(8),
            focal_distance.animate.set_value(8)
            )
        self.wait()

        udot = Dot3D(ORIGIN, radius=0.005)
        self.add(udot)

        self.play(Flash(udot, line_length=0.04, run_time=1, flash_radius=0.03))
        self.next_slide()
        

        x0s = [
            ([11/2, 3/2, 1], RED),
            ([11/2, 3/2, 1/4], GREEN),
            ([11/2, 1, 1/2], BLUE),
            ([11/2, 2, 1/2], YELLOW),
            ([5, 3/2, 1/2], PINK),
            ([6, 3/2, 1/2], PURPLE),
            ([5, 1, 1/4], GREY),
            ([6, 2, 1], ORANGE),
        ]

        dots_group = VGroup()
        for xi in x0s:
            dots_group += Dot3D(point=axes.c2p(*xi[0]), radius=0.005, color=xi[1])

        self.play(Create(dots_group))

        self.next_slide()


        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field2.nudge(dot, 0, 120)
            dot.add_updater(vector_field2.get_nudge_updater())

            self.add(path)
            grp+=path

        for p in grp:
            dots_group += p

        self.play(
            vector_field2.animate.set_opacity(0),
            run_time=1
        )


        phi, theta, focal_distance, gamma, distance_to_origin = self.camera.get_value_trackers()
        self.play(
            theta.animate.set_value((20) * DEGREES), 
            distance_to_origin.animate.set_value(12),
            focal_distance.animate.set_value(12),
            run_time=8, 
            rate_func=rate_functions.smootherstep
            )
        self.wait(10)

        self.next_slide()

        for d in dots_group:
            d.clear_updaters()

        self.play( 
            Unwrite(axes), 
            Unwrite(ax_labels),
            ShrinkToCenter(dots_group), 
            Uncreate(udot),
            run_time = 2
        )
        self.wait()
        dots_group.set_opacity(0)

        self.next_slide()


class Outro(Slide):
    def construct(self):
        thx = Text("Спасибо за внимание!").scale(1.5)

        self.play(Succession(
            Write(thx, run_time=4),
            Circumscribe(thx, color="#0066b3", run_time=2, time_width=2),
            Wiggle(thx)
        ))

        self.next_slide()

        self.play(
            Unwrite(thx, reverse=False),
            run_time=4
            )
        self.wait()
        self.next_slide()