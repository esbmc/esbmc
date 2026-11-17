# r is 1 on the path where c is false.
from helper import call


def s():
    return "a"


def i():
    return 1


c = nondet_bool()
r = call(s if c else i)
assert r == "a"
