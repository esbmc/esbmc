def f(c: bool) -> None:
    if c:
        a = 1
    else:
        a = 2
    assert type(a) == int


f(nondet_bool())
