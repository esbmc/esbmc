class Tile:
    def __init__(self, d0: int):
        self.d0: int = d0
    def __getitem__(self, sl: slice) -> "Tile":
        return Tile(sl.stop - sl.start)

t: Tile = Tile(10)
sl: slice = slice(2, 5)
sub: Tile = t[sl]
assert sub.d0 == 3
