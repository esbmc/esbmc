# KNOWNBUG until #8385: extra arguments to a method on an untracked receiver
# are dropped. CPython raises TypeError at the call below.
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
