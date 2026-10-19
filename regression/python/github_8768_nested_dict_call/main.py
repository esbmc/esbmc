def g():
    return 1


d = {"a": {"b": g}}
e = d["a"]["b"]()
assert e == 1
