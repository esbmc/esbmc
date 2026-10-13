class P:
    def size(self, k: int) -> int:
        return k

    def scale(self, k: int, m: int = 2) -> int:
        return k * m

    def keyed(self, k: int, *, j: int = 1) -> int:
        return k + j

    def total(self, *xs: int) -> int:
        return 7

    @staticmethod
    def fixed(a: int) -> int:
        return 5


class H:
    def __init__(self) -> None:
        self.p = P()
        self.items = [1, 2]


h = H()
assert h.p.size(1) == 1
assert h.p.size(k=5) == 5
assert h.p.scale(3) == 6
assert h.p.scale(3, 4) == 12
assert h.p.scale(3, m=5) == 15
assert h.p.keyed(1, j=2) == 3
assert h.p.total(1, 2) == 7
assert h.p.fixed(1) == 5
assert P.size(h.p, 4) == 4
h.items.insert(0, 3)
assert h.items[0] == 3
