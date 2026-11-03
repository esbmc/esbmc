class P:
    def __init__(self, n: int) -> None:
        self.n = n

    def plus(self, k: int) -> int:
        return self.n + k


def total(ps: list[P]) -> int:
    return ps[0].plus(1) + ps[1].plus(2)


def local() -> int:
    qs = [P(10), P(20)]
    return qs[1].plus(5)


assert total([P(1), P(2)]) == 6
assert local() == 25
