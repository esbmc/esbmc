def test_nondet_dict_empty_possible() -> None:
    """Test that empty dictionary is possible."""
    x = nondet_dict(5)
    # This should be satisfiable - empty dict is valid
    if len(x) == 0:
        assert len(x) == 0
test_nondet_dict_empty_possible()
