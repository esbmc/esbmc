class R:

    def __init__(self) -> None:
        self.r = None


def twice(n: int) -> int:
    return n * 2


def run(c: bool) -> int:
    o = R()
    if c:
        o.r = twice(2)
    if o.r is not None:
        return o.r + 1
    return 0


assert run(True) == 5
assert run(False) == 0
