def logged(f):
    def w(self, a: int, b: int) -> int:
        self.calls += 1
        return f(self, a, b)
    return w


class Acc:
    def __init__(self) -> None:
        self.calls = 0
        self.total = 0

    @logged
    def add(self, a: int, b: int) -> int:
        self.total += a + b
        return self.total


acc = Acc()
assert acc.add(1, 2) == 3
assert acc.add(3, 4) == 10
assert acc.calls == 2
