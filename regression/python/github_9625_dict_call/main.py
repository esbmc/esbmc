# dict() builds the empty dict, as {} does.
x = dict()
x["k"] = "s"
v = x.get("k")
assert v == "s"
