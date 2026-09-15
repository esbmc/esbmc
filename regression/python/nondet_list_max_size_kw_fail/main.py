# Companion to nondet_list_max_size_kw: with the keyword honoured a list of 9
# or 10 elements is reachable, so the old default bound of 8 must be falsified.
x = nondet_list(max_size=10)
assert len(x) <= 8
