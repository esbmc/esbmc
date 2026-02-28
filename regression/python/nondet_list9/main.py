def test_nondet_list_modification():
    """Test modifying elements of a nondet list."""
    x = nondet_list(10)
    __ESBMC_assume(len(x) > 0)

    x[0] = 100
    assert x[0] == 100


test_nondet_list_modification()
