# popitem() on a single-entry dict leaves it empty
def test_nondet_dict_popitem_empties_dict() -> None:
    x = nondet_dict(3, key_type=nondet_int(), value_type=nondet_int())
    __ESBMC_assume(len(x) == 1)
    k, v = x.popitem()
    assert len(x) == 0


test_nondet_dict_popitem_empties_dict()
