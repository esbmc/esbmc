class Tile:
    def __init__(self, d0: int, d1: int):
        self.d0: int = d0
        self.d1: int = d1
    def __getitem__(self, key) -> "Tile":
        return self

t: Tile = Tile(10, 20)
sub: Tile = t[slice(2, 5), slice(0, 10)]
assert sub.d0 == 10
