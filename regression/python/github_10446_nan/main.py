d = {}
d["a"] = "s"
d["a"] = float("nan")
assert not (d["a"] > 1)
assert not (d["a"] >= 1)
assert not (d["a"] < 1)
assert not (d["a"] <= 1)
assert d["a"] != 1
