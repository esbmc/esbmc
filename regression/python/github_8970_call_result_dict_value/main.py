# The value stored under "k" is the function that pick returns.
def f():
    return "a"


def g():
    return 1


def pick(c):
    if c:
        return f
    return g


c = nondet_bool()
d = {"k": pick(c)}
e = d["k"]()
assert isinstance(e, str) == c
if c:
    assert e == "a"
else:
    assert e == 1
