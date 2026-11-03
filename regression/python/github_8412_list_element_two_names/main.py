a = [1, "s", "t"]
d = a[1]
e = a[-2]
assert d is e
assert d is a[1]
assert a[0] is a[0]
