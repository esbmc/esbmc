# Companion to nondet_dict_max_size: a bound above the 8 concrete keys the old
# if-chain expansion could emit must still be reachable, so this is falsified.
d = nondet_dict(12)
assert len(d) <= 8
