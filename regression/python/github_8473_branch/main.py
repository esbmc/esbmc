def f():
    return 1


def g():
    return 2


if nondet_bool():
    b = f
else:
    b = g
if b is f:
    assert b() == 1
if b is not f:
    assert b() == 2
