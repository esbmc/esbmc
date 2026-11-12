import random


def g():
    return "a"


def pick(flag):
    if flag:
        return g


f = pick(random.random() < 0.5)
if f is not None:
    assert f() == "a"
