h = lambda y: y if y > 5 else 0.5
a = h(2)
assert isinstance(a, float)
assert a == 0.5
g = lambda y, c: y if c else (2 if y > 3 else 0.5)
assert g(1, False) == 0.5
