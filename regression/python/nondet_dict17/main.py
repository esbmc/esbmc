# Bool keys admit only False and True, so the dict cannot hold a third entry
# however large the requested bound.
d: dict[bool, int] = nondet_dict(5, key_type=nondet_bool(), value_type=nondet_int())
assert len(d) <= 2
