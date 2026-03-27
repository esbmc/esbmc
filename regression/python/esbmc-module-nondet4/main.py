from esbmc import nondet_str, __ESBMC_assume, __ESBMC_assert


def test_symbolic_string() -> None:
    s = nondet_str()
    __ESBMC_assume(len(s) <= 10)  # constrain length

    # Property: string length is within bounds
    __ESBMC_assert(len(s) >= 0 and len(s) <= 10, "String length constraint")


test_symbolic_string()
