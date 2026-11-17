# v may be the 3 or a function, so `v == 3` must compare the stored value.
import random


def f():
    return "a"


def g():
    return 1


d = {"s": f, "i": g, "n": 3}
k = "i" if random.random() < 0.5 else "n"
v = d[k]
assert v == 3 or v() == 1
