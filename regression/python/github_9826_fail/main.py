import random


def g(v):
    return v


def f(*args):
    return g(*args)


n = random.randint(0, 9)
a = f(n)
assert isinstance(a, int)
assert a != n
