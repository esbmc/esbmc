class Box:
    def __init__(self, d: dict[str, int]):
        self._error = d.get('error')

    def div(self, n: int) -> int:
        if self._error is None:
            return n
        return self._error + n


x = nondet_int()
y = nondet_int()
__ESBMC_assume(-100 < x < 100 and -100 < y < 100)
assert Box({'error': x}).div(y) == y
