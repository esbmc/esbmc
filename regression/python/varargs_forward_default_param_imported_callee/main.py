import random

from callee import func2


def func1(*args):
    return func2(*args)


if random.random() < 0.5:
    a = func1()
    assert isinstance(a, str)
    assert a == "Hello"
else:
    n = random.randint(0, 9)
    a = func1(n)
    assert isinstance(a, int)
    assert a == n
