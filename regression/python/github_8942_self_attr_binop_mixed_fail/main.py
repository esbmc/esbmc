# github #8942: an attribute set from an int and from a float argument has no
# single type, so the result of self.w * 2 is not typed from either.
class R:

    def __init__(self, w):
        self.w = w

    def area(self):
        return self.w * 2


r = R(2.5)
a = R(3).area()
assert type(a) == float
