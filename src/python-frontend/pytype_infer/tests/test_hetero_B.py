a = []

a.append("x")
a.append([])
a.append(1)
a.append("y")

assert isinstance(a[0], str)
assert isinstance(a[1], list)
assert isinstance(a[2], int)
assert isinstance(a[3], str)
