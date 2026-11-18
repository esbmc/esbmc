class Box:
    def __init__(self, d: dict[str, int]):
        self._error = d.get('error')

    def div(self, n: int) -> int:
        if self._error is None:
            return n
        return self._error + n


class Scale:
    def __init__(self, d: dict[str, float]):
        self.factor = d.get('factor')


x = nondet_int()
y = nondet_int()
__ESBMC_assume(-100 < x < 100 and -100 < y < 100)
assert Box({'error': x}).div(y) == x + y
assert Box({'other': x}).div(y) == y
assert Box({'error': 0})._error is not None
assert Scale({}).factor is None
assert Scale({'factor': 1.5}).factor == 1.5
