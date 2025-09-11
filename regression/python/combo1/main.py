from math import comb

assert comb(0, 0) == 1  # edge case: both zero
assert comb(5, 0) == 1  # choosing none
assert comb(5, 5) == 1  # choosing all
assert comb(5, 1) == 5  # simple case
assert comb(5, 4) == 5  # symmetric property
