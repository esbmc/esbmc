# helper.r is 2, not the default 1.
import helper


def f(y=0):
    return y


k = f
assert helper.r == 1
assert k(y=3) == 3
