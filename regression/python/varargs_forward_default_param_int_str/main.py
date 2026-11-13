import random


def func2(v="Hello"):
    return v


def func1(*args):
    return func2(*args)


n = random.randint(0, 9)
a = func1(n)
assert isinstance(a, int)
assert a == n
b = func1("World")
assert isinstance(b, str)
assert b == "World"
