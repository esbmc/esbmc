import random


def g():
    return 1


def func(flag):
    if flag:
        return g
    return None


a = func(random.random() > 2.0)
b = a()
assert b == 1
