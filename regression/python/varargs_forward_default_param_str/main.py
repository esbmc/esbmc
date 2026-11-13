import random


def func2(v=0):
    return v


def func1(*args):
    return func2(*args)


if random.random() < 0.5:
    a = func1()
    assert isinstance(a, int)
    assert a == 0
else:
    a = func1("World")
    assert isinstance(a, str)
    assert a == "World"
