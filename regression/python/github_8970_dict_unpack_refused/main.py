# d also holds the 5 that other brings, so it is not a dict of functions only.
def f():
    return "a"


def g():
    return 1


other = {"b": 5}
d = {"a": f, "c": g, **other}
assert d["a"]() == "a"
assert d["c"]() == 1
