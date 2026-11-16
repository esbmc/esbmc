def f(c: bool) -> None:
    if c:
        a = 1
    else:
        a = 2
    assert type(a) == float


f(nondet_bool())
