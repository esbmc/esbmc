class Tile3D:
    def __init__(self, d0: int):
        self.d0: int = d0

    def __setitem__(self, key: int, value: "Tile3D") -> None:
        assert 0 <= key
        assert key < self.d0
