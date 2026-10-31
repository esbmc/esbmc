class Pair:

    def __init__(self, a: int, b: int):
        t = (a, b)
        if a > b:
            t = (b, a)
        self._args = t

    def low(self) -> int:
        return self._args[0]


assert Pair(3, 1).low() == 3
assert Pair(1, 3).low() == 1
