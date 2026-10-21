class R:

    def __init__(self, w):
        self.w = w

    def area(self):
        return self.w * self.w


a = R(3).area()
assert isinstance(a, float)
