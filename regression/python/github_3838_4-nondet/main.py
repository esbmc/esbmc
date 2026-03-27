# Test: super().method() with nondet input
class Counter:

    def increment(self, x: int) -> int:
        return x + 1


class DoubleCounter(Counter):

    def increment(self, x: int) -> int:
        base: int = super().increment(x)
        return base + 1


c = DoubleCounter()
n: int = nondet_int()
__ESBMC_assume(n >= 0 and n <= 100)
result: int = c.increment(n)
assert result == n + 2
