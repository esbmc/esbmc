class P:
    def size(self, k: int = 3) -> int:
        return k

class H:
    def __init__(self) -> None:
        self.p = P()

def size(j: int, k: int = 5) -> int:
    return j + k

h = H()
assert h.p.size() == 3
assert size(1) == 6
