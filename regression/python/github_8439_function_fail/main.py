class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1


def run() -> int:
    o = C()
    f = o.inc
    o = C()
    f()
    return o.x


assert run() == 1
