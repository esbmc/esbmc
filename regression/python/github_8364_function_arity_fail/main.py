class P:
    def size(self, k: int) -> int:
        return k

class H:
    def __init__(self) -> None:
        self.p = P()

def size() -> int:
    return 0

h = H()
assert h.p.size(2) == 2
assert size(1) == 0
