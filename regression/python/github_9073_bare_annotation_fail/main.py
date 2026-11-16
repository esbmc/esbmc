class P:

    def __init__(self, a: int) -> None:
        self.x: tuple
        self.n = a


assert P(1).n == 2
