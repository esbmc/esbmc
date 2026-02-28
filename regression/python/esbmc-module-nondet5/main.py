from esbmc import (nondet_int, nondet_float, nondet_bool, nondet_str, nondet_list, nondet_dict,
                   __ESBMC_assume, __ESBMC_assert)


def test_nondet_all() -> None:
    """Verify all symbolic intrinsics in one unified regression."""

    # --- Integer ---
    x = nondet_int()
    __ESBMC_assume(x > -1000 and x < 1000)
    __ESBMC_assert(x > -1000 and x < 1000, "Symbolic int in bounds")

    # --- Float ---
    f = nondet_float()
    __ESBMC_assume(f > -1000.0 and f < 1000.0)
    __ESBMC_assert(f >= -1000.0 and f < 1000.0, "Symbolic float in bounds")

    # --- Boolean ---
    b = nondet_bool()
    __ESBMC_assert(b is True or b is False, "Symbolic bool is boolean")

    # --- String ---
    s = nondet_str()
    __ESBMC_assume(len(s) <= 10)
    __ESBMC_assert(len(s) >= 0 and len(s) <= 10, "Symbolic string length")

    # --- List ---
    lst = nondet_list()
    __ESBMC_assume(len(lst) <= 5)
    __ESBMC_assert(len(lst) >= 0 and len(lst) <= 5, "Symbolic list length")
    if len(lst) > 0:
        elem = lst[0]
        # Type check: allow int, float, or string
        __ESBMC_assert(
            isinstance(elem, int) or isinstance(elem, float) or isinstance(elem, str),
            "List element type valid")

    # --- Dict ---
    d = nondet_dict()
    __ESBMC_assume(len(d) <= 5)
    __ESBMC_assert(len(d) >= 0 and len(d) <= 5, "Symbolic dict size")
    if len(d) > 0:
        for k in d.keys():
            # Keys are expected to be strings
            __ESBMC_assert(isinstance(k, str), "Dict key type valid")


# Run the test
test_nondet_all()
