# Negative variant: list(filter(...)) drops the elements for which the
# predicate is false, so the materialised list has 2 elements, not 3. The
# wrong assertion below must be reported as a violation — guarding against a
# rewrite that fabricates a nondet/over-long list.


xs = [1, -2, 3, -4]
kept = list(filter(lambda x: x > 0, xs))
assert len(kept) == 3
