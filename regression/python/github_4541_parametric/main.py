class Tile:
    def __init__(self, d0: int, d1: int):
        self.d0: int = d0
        self.d1: int = d1
    def __getitem__(self, key) -> "Tile":
        sl0: slice = key[0]
        sl1: slice = key[1]
        return Tile(sl0.stop - sl0.start, sl1.stop - sl1.start)

def f(t: Tile, n: int) -> Tile:
    return t[n*4:(n+1)*4, n*8:(n+1)*8]

t: Tile = Tile(100, 200)
r: Tile = f(t, 2)
assert r.d0 == 4
