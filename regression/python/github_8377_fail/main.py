import itertools

c = itertools.chain([1, 2], [3])
assert list(c) == [1, 2]
