# float() applied to a single integer variable (not a constant)
a = 42
x = float(a)
assert x == 42.0

b = -7
y = float(b)
assert y == -7.0

c = 0
z = float(c)
assert z == 0.0
