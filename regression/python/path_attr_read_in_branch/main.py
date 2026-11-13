import random


class A:
    pass


a = A()
if random.random() < 0.5:
    a.code = 1
y = a.code if hasattr(a, "code") else 0
assert y in (0, 1)
