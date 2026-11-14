f = lambda y: y + 1
if nondet_bool():
    v = 2**60 + 1
    a = f(v)
    assert isinstance(a, float)
