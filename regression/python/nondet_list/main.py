def test_basic_nondet_list():
    """Basic nondet list with default int elements."""
    x = nondet_list(8)
    # x is a list with nondet length [0, 8] and nondet int elements
    assert len(x) >= 0

test_basic_nondet_list()
