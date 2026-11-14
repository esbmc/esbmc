import random


class A:
    pass


a = A()
if random.random() < 0.5:
    a.code = 1
if hasattr(a, "code"):
    y = a.code + 1
    assert y == 2
