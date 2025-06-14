from abc import ABC, abstractmethod

class Shape(ABC):
    @abstractmethod
    def area(self) -> float:
        """Вычислить площадь фигуры"""
        raise NotImplementedError("Subclasses must implement area()")

