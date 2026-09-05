# str keys come from a fixed table, so a str-keyed dict cannot exceed its
# length however large the requested bound (see _MAX_NONDET_STR_KEYS).
d = nondet_dict(20, key_type=nondet_str(), value_type=nondet_int())
assert len(d) <= 8
