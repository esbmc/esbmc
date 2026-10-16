a = [1, 2.5, "s", True]
b, c, d, e = a
assert isinstance(b, int)
assert isinstance(c, float)
assert isinstance(d, str)
assert isinstance(e, bool)
assert b == 1
assert b + c == 3.5
x = a[0]
assert isinstance(x, int)
