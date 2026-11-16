# A call through dict.get is not rewritten.
def f():
    return "a"


def g():
    return 1


d = {"a": f, "b": g}
assert d.get("a")() == "a"
