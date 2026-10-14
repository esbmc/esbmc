class Tile3D:
    def __init__(self, d0: int):
        self.d0: int = d0

    def __setitem__(self, key: int, value: "Tile3D") -> None:
        assert 0 <= key
        assert key < self.d0


def outer(slot: int) -> None:
    buf: Tile3D = Tile3D(2)
    src: Tile3D = Tile3D(1)

    def closure(s: int) -> None:
        buf[s] = src

    closure(slot)


outer(0)
