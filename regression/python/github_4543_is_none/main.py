class T:
    def __init__(self) -> None:
        pass

    def __getitem__(self, key) -> int:
        sl0: slice = key[0]
        assert sl0.start is None
        assert sl0.stop is None
        return 0


t: T = T()
_: int = t[:, 3:8]
