class P:
    def size(self, k: int) -> int:
        return k


class H:
    def __init__(self) -> None:
        self.p = P()


h = H()
t = (1, 2)
r = h.p.size(*t)
