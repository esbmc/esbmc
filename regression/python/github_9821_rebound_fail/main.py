# #9821, false assertion: a name rebound after reading the dict may hold another type.
x = {}
x["k"] = "s"
d = x["k"]
d = 5
x["k"] = d
e = x.pop("k")
assert e == 6
