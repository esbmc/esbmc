def test_nondet_dict_basic():
    """Test basic nondet dictionary creation and access."""
    x = nondet_dict(2)
    __ESBMC_assume(len(x) > 0)
    assert len(x) > 0

test_nondet_dict_basic()
