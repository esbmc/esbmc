# Issue #10251: the same call operand, with its type lost.
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
assert isinstance(func(g), int)
