def f(c: bool) -> None:
    a = 1
    a = 0.5
    if c:
        a = 2.5
    assert type(a) == float


f(nondet_bool())
