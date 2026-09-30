a = []

x = "x"
a.append(x)
x = []
a.append(x)
x = 1
a.append(x)
x = "y"
a.append(x)

assert isinstance(a[0], str)
assert isinstance(a[1], list)
assert isinstance(a[2], int)
assert isinstance(a[3], str)