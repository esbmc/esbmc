class R:

    def __init__(self) -> None:
        self.r = None


def twice(n: int) -> int:
    return n * 2


def run(c: bool) -> None:
    o = R()
    if c:
        o.r = twice(2)
    o.r + 1


run(True)
run(False)
