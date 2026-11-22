x = lambda x: x + 1
v = nondet_int()
a = x(v)
assert isinstance(a, int)
a2 = x(float(v % 1000))
assert isinstance(a2, int)
