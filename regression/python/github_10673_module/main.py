# Both modules pass a keyword through a function value. helper names
# __main__, so it is searched for uses of this module's names.
import helper


def f(y=0):
    return y


k = f
assert helper.r == 2
assert k(y=3) == 3
