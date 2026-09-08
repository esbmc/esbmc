# An int-keyed dict inserts in a loop, so it honours any requested bound.
d = nondet_dict(12)
assert len(d) <= 12
