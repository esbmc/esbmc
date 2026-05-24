class Tile:
    def __init__(self, d0: int, d1: int):
        self.d0: int = d0
        self.d1: int = d1


class Tile4D:
    def __init__(self, d0: int, d1: int, d2: int, d3: int):
        self.d0: int = d0
        self.d1: int = d1
        self.d2: int = d2
        self.d3: int = d3

    def __getitem__(self, key) -> "Tile":
        i: int = key[0]
        j: int = key[1]
        sl1: slice = key[2]
        sl2: slice = key[3]
        return Tile(self.d2, self.d3)


def copy(dst: Tile, src: Tile) -> None:
    assert dst.d0 == src.d0
