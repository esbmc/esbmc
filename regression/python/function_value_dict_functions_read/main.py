# A dict of functions only: a value read with a variable key is a function.
import random


def f():
    return 1


def g():
    return 2


d = {"f": f, "g": g}
k = "f" if random.random() < 0.5 else "g"
v = d[k]
assert v() == 1 or v() == 2
