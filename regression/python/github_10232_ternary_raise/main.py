import random


class P:

    @property
    def v(self):
        return 3


def h():
    return 1


p = P()
try:
    x = h() if random.random() < 0.5 else p.v()
except TypeError:
    x = 1
assert x == 1
