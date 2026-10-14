class T:
    def __getitem__(self, key) -> int:
        i: int = key[0]
        sl1: slice = key[1]
        sl2: slice = key[2]
        j: int = key[3]
        return i + (sl1.stop - sl1.start) + (sl2.stop - sl2.start) + j

t: T = T()
x: int = t[1, 2:5, 3:7, 4]
assert x == 1 + 3 + 4 + 4
