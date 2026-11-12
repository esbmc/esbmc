# Issue #10251: the value's type is lost, so isinstance() is refused.
import random

n = random.randint(0, 9)


def g():
    return n


def func(a):
    return a()


def run():
    func(lambda: n)


b = func(g)
run()
assert isinstance(b, int)
