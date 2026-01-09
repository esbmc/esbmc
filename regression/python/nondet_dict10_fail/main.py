def test_nondet_dict_key_exists() -> None:
    """Test key membership in nondet dictionary."""
    x = nondet_dict(2, key_type=nondet_int(), value_type=nondet_int())
    __ESBMC_assume(len(x) > 0)
    
    k: int = nondet_int()
    exists: bool = k in x
    assert exists == True
test_nondet_dict_key_exists()
