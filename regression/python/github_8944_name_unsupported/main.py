n = nondet_int()
if 0 <= n <= 4:
    r = 2 ** n
    assert isinstance(r, int)
