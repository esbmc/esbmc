class T:
    def __getitem__(self, key) -> int:
        sl0: slice = key[0]
        return sl0.stop - sl0.start


t: T = T()
empty: int = t[0:0, 3:8]
full: int = t[:, 3:8]
assert empty == full
