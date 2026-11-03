def f():
    return 1


def g():
    return 2


if nondet_bool():
    b = f
else:
    b = g
assert b is f
