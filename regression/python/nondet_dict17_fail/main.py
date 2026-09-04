# Companion to nondet_dict17: two bool-keyed entries are reachable, so a
# one-entry bound must be falsified.
d: dict[bool, int] = nondet_dict(5, key_type=nondet_bool(), value_type=nondet_int())
assert len(d) <= 1
