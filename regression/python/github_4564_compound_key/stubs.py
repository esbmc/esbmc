class Tile:
    def __init__(self, d0: int, d1: int):
        self.d0: int = d0
        self.d1: int = d1


class Tile3D:
    def __init__(self, d0: int, d1: int, d2: int):
        self.d0: int = d0
        self.d1: int = d1
        self.d2: int = d2

    def __getitem__(self, key) -> "Tile":
        k: int = key[0]
        sl1: slice = key[1]
        sl2: slice = key[2]
        return Tile(self.d1, self.d2)


def copy(dst: Tile, src: Tile) -> None:
    assert dst.d0 == src.d0
