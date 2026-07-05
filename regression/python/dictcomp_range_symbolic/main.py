# Dict comprehension whose iterable is range() with a non-constant bound
# (`range(n)` where n is a variable). Previously this materialised range(n)
# into a backing list that only had its size set, leaving the elements
# nondeterministic; iterating it produced wrong (nondet) keys. The
# comprehension now iterates a counter whose value IS the range element, so
# the keys are exactly 0, 1, ..., n-1. Regression for #5222 (symbolic-range
# dict-comprehension population). This case also exercises the filter clause.
n = 3
e = {i: i for i in range(n) if i % 2 == 0}
assert e[0] == 0
assert e[2] == 2
