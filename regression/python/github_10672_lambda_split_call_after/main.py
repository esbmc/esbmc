x = lambda y: y * 2
if nondet_bool():
    v = 1152921504606846977
else:
    v = 0.5
a = x(v)
assert a == v + v
assert isinstance(a, float) == isinstance(v, float)
