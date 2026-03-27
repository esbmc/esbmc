def test_nondet_list_append():
    """Test appending to a nondet list."""
    x = nondet_list(4)
    x.append(42)
    assert x[len(x) - 1] == 42

test_nondet_list_append()
