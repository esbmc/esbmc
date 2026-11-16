# #9821: re-storing a popped str keeps the dict's str value type.
x = {}
x["k"] = "s"
d = x.pop("k")
x["k"] = d
assert d[0] == "s"
assert x["k"] == "s"
