x = {}
x[2] = "s"
x[3] = "t"
e = x.popitem()
assert isinstance(e[0], int)
assert e[0] == 3
