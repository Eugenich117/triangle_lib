import math as m
from .shape import Shape

class Circle(Shape):
    def __init__(self, radius: float):
        if radius <= 0:
            raise ValueError("Радиус должен быть положительным числом.")
        self.radius = radius

    def area(self) -> float:
        return m.pi * self.radius ** 2