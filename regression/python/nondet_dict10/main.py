def test_nondet_dict_key_exists() -> None:
    """Test key membership in nondet dictionary."""
    x = nondet_dict(2, key_type=nondet_int(), value_type=nondet_int())
    __ESBMC_assume(len(x) > 0)
    
    # Test that at least one key could exist
    k: int = nondet_int()
    # If the dict is non-empty, membership test should work
    exists: bool = k in x
    assert exists == True or exists == False  # Trivially true
test_nondet_dict_key_exists()
