class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1


o = C()


def bind() -> None:
    global f
    f = o.inc


bind()
f()
assert o.x == 1
