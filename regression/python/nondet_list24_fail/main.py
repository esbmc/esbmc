# esbmc/esbmc#7575: a nondet_list in call-argument position was never expanded,
# so the callee saw a list whose elements were all the same value.
def check(z: list[int]) -> None:
    if len(z) == 2:
        assert z[0] == z[1]

check(nondet_list(3))
