# Issue #9156: a function returning a module-level name that holds a str or
# an int depending on the path.
c = nondet_bool()
if c:
    r = "a"
else:
    r = 1


def g():
    return r


z = g()
assert isinstance(z, int) != c
assert isinstance(z, str) == c
if c:
    assert z == "a"
else:
    assert z == 1
