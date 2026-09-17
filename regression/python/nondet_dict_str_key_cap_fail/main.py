# Companion to nondet_dict_str_key_cap: entries below the cap are reachable, so
# the cap is not vacuous.
d = nondet_dict(20, key_type=nondet_str(), value_type=nondet_int())
assert len(d) <= 1
