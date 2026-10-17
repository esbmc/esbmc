def f():
    return "x"


def g():
    return 42


d = {1: f, "1": g}
assert d[1]() == "x"
assert d["1"]() == 42
