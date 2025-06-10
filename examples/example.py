from shapes.circle import Circle
from shapes.triangle import Triangle
from shapes.shape import Shape
# Пример с кругом
circle = Circle(radius=5)
print(f"Площадь круга: {circle.area():.2f}")

# Пример с треугольником
triangle = Triangle(a=3, b=4, c=5)
print(f"Площадь треугольника: {triangle.area():.2f}")
print(f"Треугольник прямоугольный? - {'Да' if triangle.is_right_angled() else 'Нет'}")

# Работаем с фигурами через базовый интерфейс
shapes = [circle, triangle]
for shape in shapes:
    print(f"Площадь фигуры {type(shape).__name__}: {shape.area():.2f}")

'''пример добавления другой фигуры'''
'''для добавления другой фигуры можно создать класс и наследоваться от Shape, 
                            далее задавать абсолютно любые свойства своим фигурам'''
class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height

    def area(self):
        return self.width * self.height
"""Реализация добавления фигур не зная их типа в compile-time"""

shapes = [
    Circle(5),
    Triangle(3, 4, 5),
    Rectangle(4, 5)
]

for shape in shapes:
    print(f"Тип: {type(shape).__name__}, Площадь: {shape.area():.2f}")