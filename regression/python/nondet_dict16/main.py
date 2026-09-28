# Companion to nondet_dict16_fail: entries are independent but the requested
# bound still caps the size.
d: dict[int, int] = nondet_dict(3)
assert len(d) >= 0
assert len(d) <= 3
