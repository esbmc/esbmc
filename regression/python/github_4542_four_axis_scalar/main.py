class T:
    def __getitem__(self, key) -> int:
        return key[0] + key[1] + key[2] + key[3]

t: T = T()
x: int = t[1, 2, 3, 4]
assert x == 10
