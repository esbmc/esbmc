# #9821: a value update() stores keeps its own type when popped and re-stored.
x = {}
x["z"] = "s"
x.update({"a": 5})
d = x.pop("a")
x["k"] = d
assert x["k"] == 5
