import random


def f():
    return 1


def g():
    return 2.5


def h():
    return "a"


fs = [f, g, h]
x = fs[0]()
i = min(random.randint(0, 5), len(fs))
r = fs[i]()
assert r == x or r == 2.5 or r == "a"
