x = {}
x["k"] = "s"
v = x.pop("k")
assert v == "s"
assert len(x) == 0
