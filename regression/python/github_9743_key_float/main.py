x = dict()
x[2.0] = "s"
e = x.popitem()
assert e[0] == 2.0
assert isinstance(e[0], float)
assert e[1] == "s"
