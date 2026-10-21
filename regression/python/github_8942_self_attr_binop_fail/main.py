# github #8942: the result of self.w * self.w for an int attribute is an int.
class R:

    def __init__(self, w):
        self.w = w

    def area(self):
        return self.w * self.w


a = R(3).area()
assert isinstance(a, float)
