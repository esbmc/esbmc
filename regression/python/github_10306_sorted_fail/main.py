class C:

    def __init__(self, x: int) -> None:
        self.x = x

    def __lt__(self, other: "C") -> bool:
        return 10 // self.x < 0


sorted([C(0), C(0)])
