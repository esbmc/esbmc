class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1

    def dec(self) -> None:
        self.x -= 1


def nothing() -> None:
    pass


o = C()
for c in [True, False]:
    if c:
        f = o.inc
    else:
        f = nothing
    f()
assert o.x == 1
