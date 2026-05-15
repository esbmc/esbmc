class T:
    def __getitem__(self, key) -> int:
        sl0: slice = key[0]
        return sl0.stop


t: T = T()
y: int = t[:, 3:8]
assert y == 0
