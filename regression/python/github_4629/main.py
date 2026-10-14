from esbmc import nondet_dict, __ESBMC_assume, __ESBMC_assert

def test_symbolic_dict() -> None:
    d = nondet_dict()
    __ESBMC_assume(len(d) <= 3)

    for k in d.keys():
        __ESBMC_assert(isinstance(k, str), "All keys are strings")

test_symbolic_dict()
