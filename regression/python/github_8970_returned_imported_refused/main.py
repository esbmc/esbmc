# g is defined in another module, so call cannot be rewritten to name it.
from helper import g


def f():
    return "a"


def call(h):
    return h()


c = nondet_bool()
r = call(f if c else g)
