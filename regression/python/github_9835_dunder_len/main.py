# len(self) calls the class's own __len__ (#9835).
class C:

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def size(self) -> int:
        return len(self)


assert C(2).size() == 2
