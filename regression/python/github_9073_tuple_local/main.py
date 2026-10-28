class Pair:

    def __init__(self, a: int, b: int):
        t = (a, b)
        self._args = t

    def first(self) -> int:
        return self._args[0] // self._args[1]


x = nondet_int()
assert Pair(x, 2).first() == x // 2
