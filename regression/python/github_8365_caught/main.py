class P:
    def size(self, k: int) -> int:
        return k


class A:
    def __init__(self, x: int):
        self.x = x


def f(a: int) -> int:
    return a


def g(*, k: int) -> int:
    return k


p = P()
n = 0
try:
    f()
except TypeError:
    n += 1
try:
    p.size(1, 2)
except TypeError:
    n += 1
try:
    A(1, 2)
except TypeError:
    n += 1
try:
    g()
except TypeError:
    n += 1
assert n == 4
