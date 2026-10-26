class B:
    def __init__(self, k: int) -> None:
        self.k = k
        self.cb = self.add

    def add(self, x: int) -> int:
        return x + self.k


a = B(1)
b = B(10)
assert a.cb(2) == 3
assert b.cb(2) == 3
assert b.add(2) == 12
