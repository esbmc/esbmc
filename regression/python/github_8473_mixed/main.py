def f():
    return 1


if nondet_bool():
    b = f
else:
    b = 1
assert b is not f or b() == 1
