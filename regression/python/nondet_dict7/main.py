def test_nondet_dict_bool_to_int() -> None:
    """Test nondet dictionary with bool keys and int values."""
    x = nondet_dict(2, key_type=nondet_bool(), value_type=nondet_int())
    assert len(x) >= 0
    assert len(x) <= 2
    # Bool keys can only have at most 2 distinct values (True, False)
    __ESBMC_assume(len(x) == 2)
    assert len(x) == 2
test_nondet_dict_bool_to_int()
