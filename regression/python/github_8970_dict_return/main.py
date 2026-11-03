# make returns a dict of f and g, and the call through its value picks by key.
def f():
    return "a"


def g():
    return 1


def make():
    d = {"x": f, "y": g}
    return d


b = make()
key = "x" if nondet_bool() else "y"
r = b[key]()
if key == "x":
    assert r == "a"
else:
    assert r == 1
