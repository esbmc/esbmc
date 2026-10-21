class R:

    def __init__(self, w):
        self.w = w

    def area(self):
        return self.w * self.w


a = R(1.5).area()
assert isinstance(a, float)
assert a == 2.25
