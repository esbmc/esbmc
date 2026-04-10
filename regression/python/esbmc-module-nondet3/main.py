from esbmc import nondet_dict, __ESBMC_assume, __ESBMC_assert

def test_symbolic_dict() -> None:
    d = nondet_dict()
    __ESBMC_assume(len(d) <= 3)  # constrain size

    # If key exists, check type of value
    if "key1" in d:
        v = d["key1"]
        __ESBMC_assert(isinstance(v, int) or isinstance(v, str),
                       "Dict value type valid")

    # Property: all keys are strings
    for k in d.keys():
        __ESBMC_assert(isinstance(k, str), "All keys are strings")

test_symbolic_dict()
