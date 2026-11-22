import random


def g():
    return 1


if random.random() < 2:
    f = g
else:
    f = 3
r = f()
assert r == 1
