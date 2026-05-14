class Tile:
    def __init__(self, d0: int, d1: int):
        self.d0: int = d0
        self.d1: int = d1

    def __getitem__(self, key):
        return self


t: Tile = Tile(10, 20)
sub = t[5]
assert sub.d0 == 11
