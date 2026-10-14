class T:
    def __init__(self, d0: int):
        self.d0: int = d0
    def __getitem__(self, key) -> int:
        sl0: slice = key[0]
        sl2: slice = key[2]
        return (sl0.stop - sl0.start) + (sl2.stop - sl2.start)

t: T = T(10)
x: int = t[1:4, 5:6, 7:10]
assert x == 6
