from shapes.circle import Circle
from shapes.triangle import Triangle

# Пример с кругом
circle = Circle(radius=5)
print(f"Площадь круга: {circle.area():.2f}")

# Пример с треугольником
triangle = Triangle(a=3, b=4, c=5)
print(f"Площадь треугольника: {triangle.area():.2f}")
print(f"Треугольник прямоугольный? {'Да' if triangle.is_right_angled() else 'Нет'}")

# Работаем с фигурами через базовый интерфейс
shapes = [circle, triangle]
for shape in shapes:
    print(f"Площадь фигуры {type(shape).__name__}: {shape.area():.2f}")