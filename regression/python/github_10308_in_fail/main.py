class C:

    def __init__(self, x: int) -> None:
        self.x = x

    def __eq__(self, o: object) -> bool:
        return 10 // self.x == 0


C(1) in [C(0)]
