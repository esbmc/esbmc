def test_nondet_dict_default_size() -> None:
    """Test nondet dictionary with default size."""
    x = nondet_dict()
    assert len(x) >= 0
    assert len(x) <= 8  # Default max size


test_nondet_dict_default_size()
