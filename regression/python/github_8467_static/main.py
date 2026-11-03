class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1

    def dec(self) -> None:
        self.x -= 1


for c in [True, False]:
    o = C()
    if c:
        f = C.inc
    else:
        f = C.dec
    f(o)
    if c:
        assert o.x == 1
    else:
        assert o.x == -1
