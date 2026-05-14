class T:
    def __init__(self, d0: int, d1: int):
        self.d0: int = d0
        self.d1: int = d1
    def __getitem__(self, key) -> "T":
        sl0: slice = key[0]
        sl1: slice = key[1]
        return T(sl0.stop - sl0.start, sl1.stop - sl1.start)

t: T = T(10, 20)
sub: T = t[2:5, 3:8]
assert sub.d0 == 3
assert sub.d1 == 5
