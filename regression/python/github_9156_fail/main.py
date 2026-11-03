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
if c:
    assert z == "b"
