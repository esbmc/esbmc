from esbmc import nondet_list, __ESBMC_assume, __ESBMC_assert


def test_symbolic_list() -> None:
    lst = nondet_list()
    __ESBMC_assume(len(lst) >= 0 and len(lst) <= 5)  # constrain length

    # If list is non-empty, ensure first element is int
    if len(lst) > 0:
        x = lst[0]  # symbolic element
        __ESBMC_assert(
            isinstance(x, int) or isinstance(x, float) or isinstance(x, str),
            "First element has valid type")

    # Property: length constraint holds
    __ESBMC_assert(len(lst) >= 0 and len(lst) <= 5, "List length within bounds")


test_symbolic_list()
