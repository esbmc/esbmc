class R:
    def __init__(self) -> None:
        self.r = None


def twice(n: int) -> int:
    return n * 2


o = R()
o.r = twice(2)
assert o.r + 1 == 5
