class C:
    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1


o = C()


def g() -> None:
    o.inc()


p = o
g()
assert p.x == 0
