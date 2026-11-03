def f():
    return "a"


def h():
    return 42.5


c = nondet_bool()
if c:
    r = f()
else:
    r = h()
assert isinstance(r, str)
