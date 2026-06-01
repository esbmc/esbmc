class T:
    def __init__(self, d0: int, d1: int):
        self.d0: int = d0
        self.d1: int = d1
    def __getitem__(self, key) -> int:
        sl0: slice = key[0]
        return sl0.stop - sl0.start

t: T = T(10, 20)
x: int = t[2:5, 3:8]
assert x == 4
