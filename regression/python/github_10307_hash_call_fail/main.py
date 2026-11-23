class C:

    def __init__(self, x: int) -> None:
        self.x = x

    def __hash__(self) -> int:
        return 10 // self.x


hash(C(0))
