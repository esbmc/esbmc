# call only calls its parameter, so the str arm reaches only a call, which
# raises TypeError.
from helper import call


def f():
    return "s"


def g():
    return 7


flag = nondet_bool()
try:
    r = call(f() if flag else g)
    assert not flag and r == 7
except TypeError:
    assert flag
