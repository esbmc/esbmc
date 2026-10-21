class R:

    def __init__(self, w, h):
        self.w = w
        self.h = h

    def area(self):
        return self.w * self.w

    def perimeter(self):
        return 2 * (self.w + self.h)

    def sum(self):
        return self.w + self.h


r = R(3, 4)
a = r.area()
assert isinstance(a, int)
assert a == 9
p = r.perimeter()
assert isinstance(p, int)
assert p == 14
s = r.sum()
assert isinstance(s, int)
assert s == 7
