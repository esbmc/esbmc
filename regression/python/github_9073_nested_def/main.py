class Pair:

    def __init__(self, a: int, b: int):
        t = (a, b)

        def unused() -> int:
            return 0

        self._args = t

    def low(self) -> int:
        return self._args[0]


assert Pair(1, 3).low() == 1
