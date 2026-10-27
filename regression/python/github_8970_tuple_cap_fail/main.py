# As github_8970_tuple_cap, but the last arm asserts what is false.
def f():
    return "a"


def g():
    return 1


def k():
    return 2.5


swap = nondet_bool()
if swap:
    a, b, c = g, f, k
else:
    a, b, c = f, k, g
x = a()
y = b()
z = c()
if swap:
    assert isinstance(x, int) and isinstance(y, str) and isinstance(z, float)
    assert x == 1 and y == "a" and z == 2.5
else:
    assert isinstance(x, str) and isinstance(y, float) and isinstance(z, int)
    assert x == "a" and y == 2.5 and z == 2
