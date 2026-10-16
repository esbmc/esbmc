a = [1, "s"]
d = a[1]
a[1] = "t"
assert d is a[1]
