def pick(c: bool) -> None:
    if c:
        a = 1
    else:
        a = 0.5
    if isinstance(a, int):
        assert a == 2


pick(True)
