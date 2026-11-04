# len(self) checks the result of __len__ (#9835).
class C:

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def size(self) -> int:
        return len(self)


C(-1).size()
