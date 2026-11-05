class R:
    def __init__(self, n: int) -> None:
        self.n: int = n
        self.i: int = 0

    def __iter__(self) -> "R":
        return self

    def __next__(self) -> int:
        if self.i >= self.n:
            raise StopIteration
        self.i += 1
        return self.i

c = list(R(2))
assert c == [1, 2]
