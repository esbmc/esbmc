import random


class C:
    def __init__(self, a):
        self.a = a
        self.r = None

    def compute(self):
        self.r = self.a + self.a
        return self.r


if random.random() < 0.5:
    x = 1
else:
    x = "s"
c = C(x)
y = c.r
assert y is None
