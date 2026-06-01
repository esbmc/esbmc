class T:
    def __getitem__(self, key) -> int:
        sl0: slice = key[0]
        return sl0.start


t: T = T()
x: int = t[:, 3:8]
assert x == 0
