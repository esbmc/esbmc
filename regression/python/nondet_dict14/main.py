def test_nondet_dict_str_keys() -> None:
       x = nondet_dict(2, key_type=nondet_str(), value_type=nondet_int())
       __ESBMC_assume(len(x) > 0)
       k: str = nondet_str()
       if k in x:
           v = x[k]
           assert v == v

test_nondet_dict_str_keys()
