# A negative __len__ result on a named instance raises ValueError (#7966).
class C:
    def __init__(self, n: int):
        self.n = n
    def __len__(self) -> int:
        return self.n
c = C(-1)
k = len(c)
