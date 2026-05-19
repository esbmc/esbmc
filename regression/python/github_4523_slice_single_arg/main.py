class Buf:
    def __init__(self, x: int):
        self.x: int = x
    def __getitem__(self, key) -> int:
        return self.x

b: Buf = Buf(42)
v: int = b[slice(3)]
assert v == 42
