def test_nondet_dict_int_to_int() -> None:
    """Test nondet dictionary with int keys and int values."""
    x = nondet_dict(3, key_type=nondet_int(), value_type=nondet_int())
    assert len(x) >= 0
    assert len(x) <= 3


test_nondet_dict_int_to_int()
