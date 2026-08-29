# Materialising a zip into a list is not modelled: the resulting list has the
# wrong length and wrong elements, even when both inputs are the same length.
# Iterating the zip directly does work -- see zip_iterate_pairs.

v = list(zip([1, 2], [3, 4]))
assert len(v) == 2
assert v[0][0] == 1
