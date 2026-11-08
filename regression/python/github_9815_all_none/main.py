import random


def func(flag):
    if flag:
        return None
    return None


a = func(random.random() < 0.5)
assert a is None
