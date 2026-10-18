# 8784: a str compared with a conditional str expression.
def func():
    return "a"


def g():
    return "b"


c = nondet_bool()
f = func if c else g
r = f()
assert r == ("b" if c else "a")
