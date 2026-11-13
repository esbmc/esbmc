import random


def f():
    return "a"


def g():
    return 1


pairs = [(f, g)]
if random.random() < 0.5:
    pairs = [(g, f)]
(a, b), = pairs
r = a()
assert r == "a" or r == 1
