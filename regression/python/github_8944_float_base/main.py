n = nondet_int()
if 0 <= n <= 4:
    assert isinstance(2.0 ** n, float)
    assert type(2.0 ** n) == float
