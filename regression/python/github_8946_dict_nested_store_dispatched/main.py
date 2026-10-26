def f():
    return "a"


def g():
    return 1


d = {"a": {"b": g}}
d["a"]["b"] = f
assert d["a"]["b"]() == "a"
