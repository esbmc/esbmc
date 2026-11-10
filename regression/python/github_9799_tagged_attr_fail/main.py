import random


class C:
    def __init__(self, a):
        self.a = a

    def compute(self):
        self.r = self.a + self.a
        return self.r


if random.random() < 2.0:
    x = "s"
else:
    x = 1
c = C(x)
assert c.compute() == 2
