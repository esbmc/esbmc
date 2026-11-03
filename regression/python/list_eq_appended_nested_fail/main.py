# ret's elements are recorded by the name they were appended from, and the
# loop rebinds that name. The == fast path must compare the lists ret holds,
# not whatever rest names by the time of the comparison.
ll: list[list[int]] = [[5], [6]]
ret: list[list[int]] = []
for rest in ll:
    ret.append(rest)
assert ret == [[6], [6]]
