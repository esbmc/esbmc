x = lambda x: x + 1
v = nondet_int()
a = x(v)
assert isinstance(a, int)
v2 = float(nondet_int() % 1000)
a2 = x(v2)
assert isinstance(a2, float)
