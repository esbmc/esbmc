import random


def g():
    return "a"


f = g if random.random() < 0.5 else g()
r = f()
assert r == "a"
