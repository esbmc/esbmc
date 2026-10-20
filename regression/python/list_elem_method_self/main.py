class Counter:
    def __init__(self, n: int) -> None:
        self.n = n

    def bump(self, k: int) -> int:
        self.n += k
        return self.n


cs = [Counter(1), Counter(10)]
assert cs[1].bump(2) == 12
assert cs[1].n == 12
assert cs[0].n == 1
i = 0
assert cs[i].bump(3) == 4
assert cs[0].n == 4
