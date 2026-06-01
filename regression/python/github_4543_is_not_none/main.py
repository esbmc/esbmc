class T:
    def __init__(self) -> None:
        pass

    def __getitem__(self, key) -> int:
        sl0: slice = key[0]
        assert sl0.start is not None
        assert sl0.stop is not None
        return 0


t: T = T()
_: int = t[0:8, 3:8]
