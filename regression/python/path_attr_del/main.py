import random


class A:
    pass


a = A()
if random.random() < 0.5:
    a.code = 1
if hasattr(a, "code"):
    del a.code
assert not hasattr(a, "code")
a.code = 5
assert a.code == 5
