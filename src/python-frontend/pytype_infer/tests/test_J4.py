a = []

x = []
a.append(x)

a.append("x")

assert isinstance(a[0], list)
assert isinstance(a[1], str)