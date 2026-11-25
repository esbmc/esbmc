class C:

    def __init__(self, x: int) -> None:
        self.x = x

    def __eq__(self, o: object) -> bool:
        assert self.x != 0
        return False


xs = [C(0)]
xs.count(C(1))
