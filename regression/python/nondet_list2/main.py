def test_nondet_list_bool():
    """Nondet list with explicit int type."""
    x: list[int] = nondet_list(8, nondet_bool())
    if len(x) > 0:
        elem = x[0]
        # Each element is a nondet integer - can be any value
        assert elem == elem  # Trivially true, tests element access


test_nondet_list_bool()


def test_nondet_list_int():
    """Nondet list with explicit int type."""
    x: list[int] = nondet_list(8)
    if len(x) > 0:
        elem = x[0]
        # Each element is a nondet integer - can be any value
        assert elem == elem  # Trivially true, tests element access


test_nondet_list_int()


def test_nondet_list_float():
    """Nondet list with explicit int type."""
    x: list[int] = nondet_list(8, nondet_float())
    if len(x) > 0:
        elem = x[0]
        # Each element is a nondet integer - can be any value
        assert elem == elem  # Trivially true, tests element access


test_nondet_list_float()
