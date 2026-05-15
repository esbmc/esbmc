class Tile:
    def __init__(self, d0: int):
        self.d0: int = d0
    def __getitem__(self, sl) -> "Tile":
        return Tile(sl.stop - sl.start)

t: Tile = Tile(10)
sub: Tile = t[2:5]
assert sub.d0 == 3
