# Issue #10251: a tuple of types, with the operand's type lost.
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
assert isinstance(b, (str, int))
