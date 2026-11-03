class Counter:
    def __init__(self, n: int) -> None:
        self.n = n

    def bump(self, k: int) -> int:
        self.n += k
        return self.n


cs = [Counter(1), Counter(10)]
cs[1].bump(2)
assert cs[1].n != 12
