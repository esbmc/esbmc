import random


class P:
    def __init__(self):
        self.a = 1


def func(b):
    if b:
        return None
    return P()


p = func(False)
assert isinstance(p, P)
assert p.a == 1

q = func(random.random() < 0.5)
if q is not None:
    assert q.a == 1
else:
    assert func(True) is None
