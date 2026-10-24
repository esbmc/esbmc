# The imported call() is given, by keyword, a function returning str or int
# (#8970).
from helper import call


def s():
    return "a"


def i():
    return 1


if nondet_bool():
    f = s
else:
    f = i
r = call(g=f)
if f is i:
    assert isinstance(r, int)
