x = dict()
x[2.0] = "string"
d = x.pop(2.0)
x[2.0] = d
e = x.popitem()
assert d[0] == "t"
