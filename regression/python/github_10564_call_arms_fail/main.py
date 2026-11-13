def f():
    return 42


def g():
    return 42.5


b = nondet_bool()
c = f() if b else g()
assert isinstance(c, float)
