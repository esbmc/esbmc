class Box:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def area(self):
        return self.w * self.h


a = Box(2, 3)
b = Box(2, 1.5)
assert isinstance(a.area(), int)
assert isinstance(b.area(), float)
assert a.area() == 6
assert b.area() == 3.0
