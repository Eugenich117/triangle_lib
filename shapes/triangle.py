import math as m
from .shape import Shape

class Triangle(Shape):
    def __init__(self, a: float, b: float, c: float):
        sides = sorted([a, b, c])
        if sides[0] + sides[1] <= sides[2]:
            raise ValueError("Стороны не образуют треугольник.")
        self.a = a
        self.b = b
        self.c = c

    def area(self) -> float:
        # Формула Герона
        s = (self.a + self.b + self.c) / 2
        return m.sqrt(s * (s - self.a) * (s - self.b) * (s - self.c))

    def is_right_angled(self) -> bool:
        # Проверяем теорему Пифагора с допуском из-за погрешности вычислений
        sides = sorted([self.a, self.b, self.c])
        return m.isclose(sides[0] ** 2 + sides[1] ** 2, sides[2] ** 2, rel_tol=1e-9)