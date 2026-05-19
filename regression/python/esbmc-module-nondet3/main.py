from esbmc import (
    nondet_dict, nondet_str, nondet_int,
    __ESBMC_assume, __ESBMC_assert,
)

def test_symbolic_dict() -> None:
    d = nondet_dict(key_type=nondet_str(), value_type=nondet_int())
    __ESBMC_assume(len(d) <= 3)  # constrain size

    # Property: size constraint holds
    __ESBMC_assert(len(d) <= 3, "Dict size respects constraint")

    # Property: all keys are strings
    for k in d.keys():
        __ESBMC_assert(isinstance(k, str), "All keys are strings")

    # Property: all values are ints
    for v in d.values():
        __ESBMC_assert(isinstance(v, int), "All values are ints")

test_symbolic_dict()
