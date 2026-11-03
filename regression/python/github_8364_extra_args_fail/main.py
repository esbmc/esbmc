# CPython raises TypeError at the call below: size() takes 2 positional
# arguments but 3 were given.
class P:
    def size(self, k: int) -> int:
        return k

class H:
    def __init__(self) -> None:
        self.p = P()

def size() -> int:
    return 0

h = H()
r = h.p.size(1, 2)
