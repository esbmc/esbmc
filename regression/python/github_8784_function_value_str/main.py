# The result of a function chosen by a conditional, compared with a
# conditional str.
def func():
    return "a"


def g():
    return "b"


c = nondet_bool()
f = func if c else g
r = f()
assert r == ("a" if c else "b")
