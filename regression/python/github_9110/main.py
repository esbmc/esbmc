d: dict[int, float] = {}
i = 2
d[i] = "s"
x = d[i]
assert not isinstance(x, float)
assert x == "s"
