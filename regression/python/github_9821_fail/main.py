# #9821, false assertion: re-storing a popped str keeps the str type.
x = {}
x["k"] = "s"
d = x.pop("k")
x["k"] = d
assert d[0] == "s"
assert x["k"] == "t"
