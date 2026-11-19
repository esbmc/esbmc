def pick(c: bool):
    if c:
        return 1
    return 0.5


def f(c: bool) -> None:
    a = 1 if c else 0.5
    b = pick(c)
    assert a + b >= 2


f(nondet_bool())
