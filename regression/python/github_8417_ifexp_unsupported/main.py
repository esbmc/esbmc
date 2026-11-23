def f(c: bool) -> None:
    a = 1 if c else 0.5
    if isinstance(a, int):
        assert a == 1
    else:
        assert a == 0.5


f(nondet_bool())
