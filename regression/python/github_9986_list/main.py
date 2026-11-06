class R:

    def __init__(self, n: int) -> None:
        self.n: int = n

    def __iter__(self) -> "R":
        return self

    def __next__(self) -> int:
        if self.n <= 0:
            raise StopIteration
        self.n -= 1
        return self.n


class Box:

    def R(self) -> int:
        return 0


assert list(R(2)) == [1, 0]
