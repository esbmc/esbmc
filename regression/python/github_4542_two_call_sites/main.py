class T:
    def __getitem__(self, key) -> int:
        i: int = key[0]
        sl: slice = key[1]
        return i + sl.stop

t: T = T()
a: int = t[3, 1:5]
b: int = t[7, 2:9]
assert a == 8
assert b == 16
