# pick returns a conditional expression, and its result is called twice.
def f():
    return "a"


def g():
    return 1


def pick(c):
    return f if c else g


c = nondet_bool()
a = pick(c)
b = pick(not c)
r = a()
s = b()
assert isinstance(r, str) != isinstance(s, str)
assert isinstance(r, str) == c
