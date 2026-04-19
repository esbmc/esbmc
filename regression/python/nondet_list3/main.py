def test_nondet_list_append_bool():
    """Test appending to a nondet list."""
    x = nondet_list(4, nondet_bool())
    original_len = len(x)
    x.append(42)
    assert len(x) == original_len + 1

test_nondet_list_append_bool()

def test_nondet_list_append():
    """Test appending to a nondet list."""
    x = nondet_list(4)
    original_len = len(x)
    x.append(42)
    assert len(x) == original_len + 1

test_nondet_list_append()

def test_nondet_list_append_float():
    """Test appending to a nondet list."""
    x = nondet_list(4, nondet_float())
    original_len = len(x)
    x.append(42)
    assert len(x) == original_len + 1

test_nondet_list_append_float()
