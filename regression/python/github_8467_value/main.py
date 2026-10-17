class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self, d: int) -> int:
        self.x += d
        return self.x

    def dec(self, d: int) -> int:
        self.x -= d
        return self.x


def pick(c: bool, o: C) -> int:
    if c:
        f = o.inc
    else:
        f = o.dec
    y = f(2)
    assert y == o.x
    return f(3)


assert pick(True, C()) == 5
assert pick(False, C()) == -5
