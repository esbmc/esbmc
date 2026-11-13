import random


def f():
    return 1


def g():
    return 2.5


def h():
    return "a"


fs = [f, g, h]
x = fs[0]()
y = fs[1]()
z = fs[2]()
i = random.randint(0, 2)
r = fs[i]()
assert x == 1 and y == 2.5 and z == "b"
assert r == x or r == y or r == z
