class C:

    def __init__(self, x: int) -> None:
        self.x = x

    def __hash__(self) -> int:
        return self.x


c = C(5)
assert hash(c) == 5
assert hash(C(-1)) == -2
