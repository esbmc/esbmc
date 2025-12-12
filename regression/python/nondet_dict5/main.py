def test_nondet_dict_basic() -> None:
    """Test basic nondet dictionary creation and access."""
    x = nondet_dict(2)
    assert len(x) >= 0
    assert len(x) <= 2

test_nondet_dict_basic()
