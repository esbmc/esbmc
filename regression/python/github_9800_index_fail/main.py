try:
    from esbmc import nondet_int
except ImportError:

    def nondet_int() -> int:
        return 2


i = nondet_int()
assert type(1) is not [float, str][i]
