import random
from typing import Any


class C:
    def __init__(self, a):
        self.a = a
        self.twice = a + a
        self.unset: Any

    def compute(self):
        self.r = self.a + self.a
        return self.r

    def copy(self):
        self.b = self.a
        return self.b


if random.random() < 0.5:
    x = 1
else:
    x = "s"
c = C(x)
assert c.compute() == x + x
assert c.twice == x + x
assert c.copy() == x
