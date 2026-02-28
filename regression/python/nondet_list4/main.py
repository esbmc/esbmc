def test_list_size_nonnegative():
    """Verify list size is always non-negative."""
    x = nondet_list(5)
    assert len(x) >= 0


test_list_size_nonnegative()


def test_list_size_bounded():
    """Verify list size is bounded by max length."""
    x = nondet_list(6)
    # Default --nondet-list-length=8
    assert len(x) <= 8


test_list_size_bounded()


def test_list_can_be_empty():
    """Verify empty list is a valid outcome."""
    x = nondet_list(0)
    if len(x) == 0:
        # Empty list is valid
        assert True


test_list_can_be_empty()


def test_list_can_have_elements():
    """Verify non-empty list is a valid outcome."""
    x = nondet_list(2)
    # Assume we have at least one element for this test
    __ESBMC_assume(len(x) > 0)
    assert x[0] is not None


test_list_can_have_elements()
