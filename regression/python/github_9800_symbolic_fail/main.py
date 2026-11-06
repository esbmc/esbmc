try:
    from esbmc import nondet_int
except ImportError:

    def nondet_int() -> int:
        return 0


i = nondet_int()
if -3 <= i < 3:
    assert type(2.5) is [int, float, str][i]
