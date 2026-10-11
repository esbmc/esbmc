class P:
    def size(self, k: int) -> int:
        return k


class H:
    def __init__(self) -> None:
        self.p = P()


h = H()
r = h.p.size()
