f = lambda y: y + 1
if nondet_bool():
    v = 2**60 + 1
    a = f(v)
    assert isinstance(a, int)
    assert a == 2**60 + 2
else:
    w = 2.5
    b = f(w)
    assert isinstance(b, float)
    assert b == 3.5
