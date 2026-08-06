# `None in <list>` is False in CPython, but the None path answers True for any
# operator that is not Eq/Is, so `x in nums` comes back True -- and `x not in
# nums` is True as well, which cannot both hold. Separate from the arithmetic
# TypeError of GitHub #6260, which is fixed; this is the remaining gap in the
# same code path.
x = None
nums = [1, 2, 3]

assert not (x in nums)
