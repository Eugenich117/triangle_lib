import unittest
from shapes.circle import Circle
from shapes.triangle import Triangle

class TestCircle(unittest.TestCase):
    def test_area(self):
        c = Circle(2)
        expected_area = 3.141592653589793 * 4
        self.assertAlmostEqual(c.area(), expected_area)

    def test_invalid_radius(self):
        with self.assertRaises(ValueError):
            Circle(0)

class TestTriangle(unittest.TestCase):
    def test_area(self):
        t = Triangle(3, 4, 5)
        expected_area = 6.0
        self.assertAlmostEqual(t.area(), expected_area)

    def test_is_right_angled(self):
        t = Triangle(3, 4, 5)
        self.assertTrue(t.is_right_angled())

    def test_not_right_angled(self):
        t = Triangle(3, 4, 6)
        self.assertFalse(t.is_right_angled())

    def test_invalid_triangle(self):
        with self.assertRaises(ValueError):
            Triangle(1, 2, 3)  # Не треугольник

if __name__ == "__main__":
    unittest.main()
