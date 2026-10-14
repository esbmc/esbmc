class Tile:
    def __init__(self, d0: int, d1: int):
        self.d0: int = d0
        self.d1: int = d1


class Container:
    def __init__(self) -> None:
        self.value: Tile = Tile(0, 0)

    def __setitem__(self, key, val: Tile) -> None:
        self.value = val


def make_tile(d0: int, d1: int) -> Tile:
    return Tile(d0, d1)
