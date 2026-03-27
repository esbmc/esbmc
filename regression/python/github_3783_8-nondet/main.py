# popitem() removes exactly one entry: dict size decreases by 1
def test_nondet_dict_popitem_key_removed() -> None:
    x = nondet_dict(3, key_type=nondet_int(), value_type=nondet_int())
    __ESBMC_assume(len(x) > 0)
    original_size = len(x)
    k, v = x.popitem()
    assert len(x) == original_size - 1


test_nondet_dict_popitem_key_removed()
