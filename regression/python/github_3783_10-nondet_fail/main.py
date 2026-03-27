# popitem() on a possibly-empty nondet dict raises KeyError
def test_nondet_dict_popitem_empty_keyerror() -> None:
    x = nondet_dict(2, key_type=nondet_int(), value_type=nondet_int())
    # No assumption on len(x): empty dict is reachable -> KeyError
    k, v = x.popitem()


test_nondet_dict_popitem_empty_keyerror()
