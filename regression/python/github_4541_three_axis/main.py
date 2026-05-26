class Tile:
    def __init__(self, d0: int, d1: int, d2: int):
        self.d0: int = d0
        self.d1: int = d1
        self.d2: int = d2
    def __getitem__(self, key) -> "Tile":
        sl0: slice = key[0]
        sl1: slice = key[1]
        sl2: slice = key[2]
        return Tile(sl0.stop - sl0.start,
                    sl1.stop - sl1.start,
                    sl2.stop - sl2.start)

def f(t: Tile) -> Tile:
    return t[1:4, 2:7, 3:9]

t: Tile = Tile(10, 20, 30)
r: Tile = f(t)
assert r.d0 == 3
assert r.d1 == 5
assert r.d2 == 6
