def test_nondet_dict_empty_possible() -> None:
    """Test that empty dictionary is possible."""
    x = nondet_dict(5)
    if len(x) == 0:
        assert len(x) == 1


test_nondet_dict_empty_possible()
