class T:
    def __getitem__(self, key) -> int:
        return key[0]

t: T = T()
b: int = t[7, 9]
assert b == 7
