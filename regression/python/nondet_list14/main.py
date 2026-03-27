def test_nondet_list_float_modification():
    """Test modifying elements of a nondet list."""
    x = nondet_list(4, nondet_float())
    __ESBMC_assume(len(x) > 0)

    x.append(10.5)
    assert x[len(x) - 1] >= 10.5

test_nondet_list_float_modification()
