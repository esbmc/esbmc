class Tile:
    def __init__(self, d0: int):
        self.d0: int = d0
    def __getitem__(self, key) -> "Tile":
        return self

t: Tile = Tile(10)
sub: Tile = t[2:5]
assert sub.d0 == 99
