class T:
    def __init__(self) -> None:
        pass
    def __getitem__(self, key) -> int:
        return 42

t: T = T()
a: int = t[2:5, 3:8]
b: int = t[0, 1]
assert a == 42
assert b == 42
