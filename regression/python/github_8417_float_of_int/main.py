x = nondet_int()
a = float(x % 10)
assert isinstance(a, float)
assert not isinstance(a, int)
