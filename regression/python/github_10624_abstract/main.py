from abc import ABC, abstractmethod


class Shape(ABC):
    @abstractmethod
    def area(self):
        pass


class Rect(Shape):
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def area(self):
        return self.w * self.h


a = Rect(2, 3)
b = Rect(2.5, 2)
assert isinstance(a, Shape)
assert isinstance(b, Shape)
assert isinstance(a.area(), int)
assert isinstance(b.area(), float)
assert b.area() == 5.0
