x = True
d = {}
d["a"] = x
assert isinstance(d["a"], bool)
assert d["a"]
