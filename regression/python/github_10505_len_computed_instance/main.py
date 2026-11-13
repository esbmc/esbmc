# len() of an expression that evaluates to a class instance (#10505)
import array


class C:

    def __init__(self, n: int) -> None:
        self.n = n

    def __len__(self) -> int:
        return self.n

    def __mul__(self, k: int) -> "C":
        return C(self.n * k)


def make(n: int) -> C:
    return C(n)


assert len(array.array("b", [1, 2]) * 2) == 4
assert len(array.array("b", [1, 2])) == 2
assert len(C(2) * 3) == 6
assert len(make(4)) == 4
