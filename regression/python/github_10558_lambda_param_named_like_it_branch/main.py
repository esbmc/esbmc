x = lambda x: x + 1
v = nondet_int()
if nondet_bool():
    a = x(v)
    assert a - 1 == v
    assert isinstance(a, int)
else:
    a2 = x(float(v % 1000))
    assert isinstance(a2, float)
    assert a2 == float(v % 1000) + 1.0
