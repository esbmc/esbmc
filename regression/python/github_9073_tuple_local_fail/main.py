class Pair:

    def __init__(self, a: int, b: int):
        t = (a, b)
        self._args = t

    def first(self) -> int:
        return self._args[0] // self._args[1]


Pair(nondet_int(), 0).first()
