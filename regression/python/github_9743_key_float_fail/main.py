x = dict()
x[2.0] = "s"
e = x.popitem()
assert e[0] == 3.0
