import random


def func2(v="Hello"):
    return v


def func1(*args):
    return func2(*args)


a = func1()
assert isinstance(a, str)
assert a == "Hello"
n = random.randint(0, 9)
b = func1(n)
assert isinstance(b, str)
