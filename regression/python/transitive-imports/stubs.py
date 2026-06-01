class Tile:
    def __init__(self, d0: int, d1: int):
        self.d0: int = d0
        self.d1: int = d1


def make_tile(d0: int, d1: int) -> Tile:
    assert d0 > 0
    assert d1 > 0
    return Tile(d0, d1)


def check_shape(a: Tile, b: Tile) -> None:
    assert a.d0 == b.d0
    assert a.d1 == b.d1
