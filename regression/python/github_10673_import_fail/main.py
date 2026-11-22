# a(x=2) is 2, not the default 1.
import random


def g(x=1):
    return x


a = g
assert a(x=2) == 1
