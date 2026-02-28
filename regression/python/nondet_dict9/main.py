def test_nondet_dict_float_values() -> None:
    """Test nondet dictionary with float values."""
    x = nondet_dict(2, key_type=nondet_int(), value_type=nondet_float())
    __ESBMC_assume(len(x) > 0)

    # Test value access with nondet key
    k: int = nondet_int()
    if k in x:
        v = x[k]
        # Float value comparisons
        assert v == v  # Reflexivity


test_nondet_dict_float_values()
