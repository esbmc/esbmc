def test_nondet_list_bool_modification():
    """Test modifying elements of a nondet list."""
    x = nondet_list(4, nondet_bool())
    __ESBMC_assume(len(x) > 0)

    x.append(True)
    assert x[len(x) - 1] == True


test_nondet_list_bool_modification()
