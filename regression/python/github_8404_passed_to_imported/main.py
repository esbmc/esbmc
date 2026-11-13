# The imported call() calls the function it is given, which returns str or
# int. call() reads no name of its module, so a copy of it is dispatched.
from helper import call


def s():
    return "a"


def i():
    return 1


if nondet_bool():
    f = s
else:
    f = i
r = call(f)
if f is i:
    assert isinstance(r, int)
