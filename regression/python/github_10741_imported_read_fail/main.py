# show reads its parameter as a value, so the conditional stays refused.
from helper import show


def f():
    return "s"


def g():
    return 7


r = show(f() if nondet_bool() else g)
assert r == 7
