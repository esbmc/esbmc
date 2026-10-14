class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1

    def dec(self) -> None:
        self.x -= 1


def shadow() -> int:
    f = 3
    return f


for c in [True, False]:
    o = C()
    p = C()
    if c:
        f = o.inc
    else:
        f = p.dec
    f()
    if c:
        assert o.x == 1 and p.x == 0
    else:
        assert o.x == 0 and p.x == -1
assert shadow() == 3
