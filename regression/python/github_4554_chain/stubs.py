class Tile:
    def __init__(self, d0: int, d1: int):
        self.d0: int = d0
        self.d1: int = d1
    def __getitem__(self, key) -> "Tile":
        sl0: slice = key[0]
        sl1: slice = key[1]
        return Tile(sl0.stop - sl0.start, sl1.stop - sl1.start)
