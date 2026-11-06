import random


class A:
    pass


a = A()
code = 0
if random.random() < 0.5:
    code = 7
    a.code = code
assert getattr(a, "code", 0) == code
