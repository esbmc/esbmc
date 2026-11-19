def pick(c: bool):
    if c:
        return 1
    return 0.5


def f(c: bool) -> None:
    b = pick(c)
    if type(b) == int:
        assert b == 1
    else:
        assert b == 0.5


f(nondet_bool())
