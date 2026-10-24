def f1():
    return 42


def f2():
    return 7


d = {"a": {"b": f1}}
d["a"] = {"b": f2}
assert d["a"]["b"]() == 42
