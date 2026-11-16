# #10075: a float-or-None call result at module level.
def h():
    return 1.5


def g():
    return None


if nondet_bool():
    r = h()
else:
    r = g()
assert r != 1.5
