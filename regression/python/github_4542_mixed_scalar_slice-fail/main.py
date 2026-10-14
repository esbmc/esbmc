class T:
    def __getitem__(self, key) -> int:
        return key[0]

t: T = T()
x: int = t[2, 3:8, 5:10]
assert x == 3
