# After extend the list has 4 elements, not 5; asserting otherwise must fail.
a = [1, 2]
a.extend([3, 4])
assert len(a) == 5
