a, b = {1: 2.0}, {2: 3.0}
a, b = b, a
assert 2 in a
assert 1 in b
x, y = 1, 2
x, y = y, x + y
assert x == 2
assert y == 3
