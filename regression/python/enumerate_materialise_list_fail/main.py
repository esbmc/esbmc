# Negative counterpart of enumerate_materialise_list. The length assertion
# holds, so the failure has to land on the index, which counts from 0. Pinning
# the line is what makes this bite: before the fix the failure landed on the
# construction, not the assertion.

e = list(enumerate([7, 8]))
assert len(e) == 2
assert e[1][0] == 0
