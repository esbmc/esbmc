class C:

    def __init__(self) -> None:
        self.x = 0

    def inc(self) -> None:
        self.x += 1

    def swap_and_inc(self) -> int:
        f = self.inc
        self = C()
        f()
        return self.x


def run() -> int:
    o = C()
    f = o.inc
    o = C()
    f()
    return o.x


assert run() == 0
assert C().swap_and_inc() == 0
