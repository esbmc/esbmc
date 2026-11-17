# call of helper calls a function that returns a str or an int, by the path.
from helper import call


def s():
    return "a"


def i():
    return 1


c = nondet_bool()
r = call(s if c else i)
assert (r == "a") == c
assert isinstance(r, int) != c
