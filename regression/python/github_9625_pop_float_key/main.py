x = dict()
x[2.0] = "string"
d = x.pop(2.0)
assert d[0] == "s"
