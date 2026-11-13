d = {}
d["a"] = "s"
d["a"] = float("nan")
assert d["a"] > 1 or d["a"] <= 1
