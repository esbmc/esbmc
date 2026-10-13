class P:
    def size(self, k: int) -> int:
        return k

    def run(self) -> int:
        return self.size(1, 2)


class H:
    def __init__(self) -> None:
        self.p = P()


h = H()
h.p.run()
