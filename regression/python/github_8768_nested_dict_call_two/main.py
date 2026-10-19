def g():
    return 1


def h():
    return 2


d = {"a": {"b": g, "c": h}}
assert d["a"]["b"]() == 1
assert d["a"]["c"]() == 2
