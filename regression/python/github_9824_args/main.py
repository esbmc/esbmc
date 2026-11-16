import random


def g(x: int) -> int:
    return x + 1


def func(flag: bool):
    if not flag:
        return "s"
    return g


f = random.random() < 0.5
h = func(f)
try:
    r = h(2)
    assert f and r == 3
except TypeError:
    assert not f
