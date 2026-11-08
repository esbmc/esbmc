import random


def g():
    return 1


def func(flag):
    if flag:
        return g
    return None


a = func(random.random() < 2.0)
if a is not None:
    b = a()
    assert b == 1
