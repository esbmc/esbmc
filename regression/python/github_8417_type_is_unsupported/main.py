def f(c: bool) -> None:
    if c:
        a = 1
    else:
        a = 0.5
    if type(a) is int:
        assert a == 2


f(nondet_bool())
