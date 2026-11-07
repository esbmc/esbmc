import random


def f3(a, x):
    return a(x)


def f2(b, x):
    return f3(b, x)


e = f2(lambda y: y > 0, random.randint(0, 9))
assert isinstance(e, bool)
assert f2(lambda y: y + 1, 3) == 4
