# #9821, false assertion: a value update() stores keeps its own type.
x = {}
x["z"] = "s"
x.update({"a": 5})
d = x.pop("a")
x["k"] = d
assert x["k"] == 6
