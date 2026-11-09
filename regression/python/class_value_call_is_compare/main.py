import random


class C:
    pass


class D:
    pass


class E:
    pass


cls = C if random.random() < 0.5 else D
c = cls()
assert cls is C or cls is D
if cls is C:
    assert isinstance(c, C)
if D == cls:
    assert isinstance(c, D)
assert cls is not E
assert cls != E
