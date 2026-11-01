# An empty dict filled by item assignment: get() takes the type of the stored str.
x = {}
x["k"] = "s"
v = x.get("k")
assert v == "s"
