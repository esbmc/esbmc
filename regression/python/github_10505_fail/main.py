# #10505: len() of an instance the AST names no class for calls __len__.
import array


class C:

    def __init__(self, n: int) -> None:
        self.n = n

    def __mul__(self, k: int) -> "C":
        return C(self.n * k)

    def __len__(self) -> int:
        return self.n


def make() -> C:
    return C(3)


assert len(array.array("b", [1, 2]) * 2) == 4
assert len(C(2) * 3) == 5
assert len(make()) == 3
