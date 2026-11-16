import random


class C:
    pass


class D:
    pass


cls = C if random.random() < 0.5 else D
c = cls()
if cls is not D:
    assert isinstance(c, D)
else:
    assert isinstance(c, C)
