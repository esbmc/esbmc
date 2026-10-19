def g():
    return 1


d = {"a": {"b": g}}
assert d["a"]["b"]() == 2
