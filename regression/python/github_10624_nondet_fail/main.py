class Rect:
    def __init__(self, w, h):
        self.w = w
        self.h = h

    def area(self):
        return self.w * self.h


w = nondet_int()
h = nondet_int()
a = Rect(w % 100, h % 100)
b = Rect(w % 100, float(h % 100))
assert isinstance(a.area(), int)
assert isinstance(b.area(), float)
assert b.area() != a.area()
