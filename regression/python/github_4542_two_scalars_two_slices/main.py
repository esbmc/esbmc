class T:
    def __getitem__(self, key) -> int:
        return key[0] + key[1]

t: T = T()
x: int = t[2, 0, :, :]
assert x == 2
