x = lambda x: x + 1
if nondet_bool():
    v = 1152921504606846977
    a = x(v)
else:
    v = float(nondet_int() % 1000)
    a = x(v)
assert isinstance(a, int) == isinstance(v, int)
assert a - 1 == v
