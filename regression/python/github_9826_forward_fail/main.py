import random


def f(x):
    return g(x)


def g(v):
    return v


n = random.randint(0, 9)
a = f(n)
assert isinstance(a, int)
assert a == n + 1
