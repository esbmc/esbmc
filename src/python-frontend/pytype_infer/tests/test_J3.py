a = []
a.append("x")

x = []
a.append(x)

assert isinstance(a[0], str)
assert isinstance(a[1], list)

