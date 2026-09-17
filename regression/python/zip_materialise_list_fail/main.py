# Negative counterpart of zip_materialise_list. The length assertion holds --
# zip truncates to the shorter input -- so the run has to get past it and fail
# on the element, which is 5. Pinning the line is what makes this bite: before
# the fix the failure landed on the construction, not the assertion.

xs = [1, 2, 3]
ys = [4, 5]
w = list(zip(xs, ys))
assert len(w) == 2
assert w[1][1] == 4
