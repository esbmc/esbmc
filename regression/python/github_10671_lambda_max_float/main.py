h = lambda y: max(y, 0.5)
a = h(0)
assert isinstance(a, float)
assert a == 0.5
assert h(2.5) == 2.5
