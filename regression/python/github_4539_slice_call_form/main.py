class T:
    def __init__(self, d0: int):
        self.d0: int = d0
    def __getitem__(self, key) -> int:
        sl1: slice = key[1]
        return sl1.stop - sl1.start

t: T = T(10)
x: int = t[slice(2, 5), slice(0, 7)]
assert x == 7
