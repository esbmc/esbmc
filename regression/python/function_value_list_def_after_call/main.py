import random


def f():
    return 1


def h():
    return "a"


fs = [f, h]
i = random.randint(0, 1)
r = fs[i]()
assert r == 1 or r == "a"


def k():
    return True


assert k()
