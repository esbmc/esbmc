d: dict[int, float] = {}
d[2] = "wrong-type"
x = d[2]
assert isinstance(x, float)
