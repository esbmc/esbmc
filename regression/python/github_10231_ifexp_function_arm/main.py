import random


def g():
    return "a"


c = random.random() < 0.5
f = g if c else g()
if c:
    r = f()
    assert r == "a"
