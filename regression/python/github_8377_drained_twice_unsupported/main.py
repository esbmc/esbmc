# A second drain of one chain iterator yields nothing, so the name is not
# replaced by a list and the program is refused (#8377).
import itertools

c = itertools.chain([1, 2], [3])
assert list(c) == [1, 2, 3]
assert list(c) == []
