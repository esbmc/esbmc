def apply(f, x: int) -> int:
    return f(x)


class S:
    def __init__(self) -> None:
        self.k: int = 10

    def run(self) -> int:
        return apply(lambda p: p + 1, 1)

    def read(self, n: int) -> int:
        m = 5
        f = lambda p: p + self.k + m + n
        return f(1)

    @staticmethod
    def twice() -> int:
        return apply(lambda p: p * 2, 4)


s = S()
assert s.run() == 2
assert s.read(100) == 116
assert S.twice() == 9
