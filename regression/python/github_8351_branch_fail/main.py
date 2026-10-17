# Calling a function variable bound in either branch types the result from
# the functions it may hold (#8351).
def f1() -> int:
    return 1


def f2() -> int:
    return 2


if nondet_bool():
    g = f1
else:
    g = f2
r = g()
assert r == 1
