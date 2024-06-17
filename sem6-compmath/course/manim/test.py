from manim import *
from manim_slides import Slide
import numpy as np

# config.quality = "example_quality"
config.quality = "high_quality"

class Present(Slide):
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

        self.play(Write(arr), Write(arr2), Write(arr3), Write(arr4))
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


class Present2(Slide, MovingCameraScene):
    def construct(self):
        ksi1, ksi2, ksi3 = 10, 8, 6
        a12, a13, a23 = 6, 2, 0.5
        k12, k13, k23 = 4, 1, 0.5
        def right_x1(x):
            x= x*10
            return np.array([
                (ksi2 - a23 * x[1]) * x[0],
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
                (ksi2 + k12 * a12 * x[0]) * x[1],
                0
            ])/100
        

        self.camera.frame.save_state()

        vector_field1 = ArrowVectorField(
            right_x1,
            # padding=0.01,
            color=BLUE,
            x_range=[0,8,0.5],
            y_range=[0,6,0.5],
        )

        # Axes._origin_shift = lambda *x: 0
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

        self.play(Create(dots_group))
        self.next_slide()

        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field1.nudge(dot, 0.01, 60)
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
            Transform(vector_field1, vector_field2), 
            Transform(axes, axes2),
            Transform(x1l, MathTex("x_1").move_to(x1l) ))
        self.play(Indicate(x20t))

        self.next_slide()

        dots_group = VGroup()
        dots_group += Dot(color=GREEN).move_to(axes2.c2p(2,1,0))
        dots_group += Dot(color=BLUE).move_to(axes2.c2p(2,1.5,0))
        dots_group += Dot(color=YELLOW).move_to(axes2.c2p(2,2,2,0))
        dots_group += Dot(color=PINK).move_to(axes2.c2p(2,2.5,0))

        dots_group += Dot(color=WHITE).move_to(axes2.c2p(2.5,5,0))

        self.play(Create(dots_group))
        self.next_slide()

        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field2.nudge(dot, 0.01, 60)
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
            # padding=0.01,
            color=BLUE,
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
            Transform(vector_field1, vector_field3), 
            Transform(axes, axes3),
            Transform(x2l, MathTex("x_2").move_to(x2l)))
        self.play(Indicate(x30t))

        self.next_slide()

        dots_group = VGroup()
        dots_group += Dot(color=GREEN).move_to(axes.c2p(5,5,0))
        dots_group += Dot(color=BLUE).move_to(axes.c2p(10,5,0))
        dots_group += Dot(color=YELLOW).move_to(axes.c2p(15,5,0))
        dots_group += Dot(color=PINK).move_to(axes.c2p(20,5,0))

        self.play(Create(dots_group))
        self.next_slide()

        grp = VGroup()
        for dot in dots_group:
            path = TracedPath(dot.get_center, stroke_color=dot.color)
            vector_field3.nudge(dot, 0.01, 120)
            dot.add_updater(vector_field3.get_nudge_updater())

            self.add(path)
            grp+=path

        dots_group += grp
        self.wait(1)
        self.next_slide()
        for dot in dots_group:
            dot.clear_updaters()

        # vector_field2.set_opacity(0)
        # vector_field2.set_opacity(0)
        # axes.set_opacity(0)
        self.play(
            FadeOut(x30t, shift=DOWN), 
            FadeOut(vector_field1, scale=0.5), 
            FadeOut(axes, scale=0.5), 
            FadeOut(ax_labels, scale=0.5), 
            ShrinkToCenter(dots_group))

        self.next_slide()







