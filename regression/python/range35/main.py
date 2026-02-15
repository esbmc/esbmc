# Test direct len(range(...)) call
x = len(range(10))
assert x == 10

y = len(range(5, 15))
assert y == 10

z = len(range(0, 20, 2))
assert z == 10
