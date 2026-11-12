d = {}
d["a"] = "s"
d["a"] = True
assert d["a"] == 1.0
assert d["a"] < 1.5
assert d["a"] > 0.5
