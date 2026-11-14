# random binds a name `a` of its own, which is not this module's a.
import random


def g(x=1):
    return x


a = g
assert a(x=2) == 2
