# Unpacking assignment
x, y = 1, 2
assert x == 1
assert y == 2
x += 5
y *= 3
assert x == 6
assert y == 6

# Nested multiple assignment
(c, d) = (e, f) = (1, 2)
assert c == 1
assert d == 2
assert e == 1
assert f == 2
c += 10
d += 20
e += 30
f += 40
assert c == 11
assert d == 22
assert e == 31
assert f == 42
