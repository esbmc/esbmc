import random


def f3(a, x):
    return a(x)


def f2(b, x):
    return f3(b, x)


x = random.randint(1, 9)
e = f2(lambda y: y > 0, x)
assert isinstance(e, bool)
assert not e
