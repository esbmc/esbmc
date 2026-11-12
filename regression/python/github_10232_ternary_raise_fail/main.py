import random


class P:

    @property
    def v(self):
        return 3


def h():
    return 1


p = P()
x = h() if random.random() < -1 else p.v()
assert x == 1
