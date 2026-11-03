# chain and repeat with a count, drained at once by a builtin (#8377).
import itertools
from itertools import chain as ch, repeat

c = itertools.chain([1, 2], [3])
assert list(c) == [1, 2, 3]

xs = [4, 5]
assert sum(ch(xs, (6, ), range(2))) == 16
assert sorted(itertools.chain([3], [1, 2])) == [1, 2, 3]
assert max(itertools.chain(xs, [9])) == 9
assert list(ch("ab", "c")) == ["a", "b", "c"]
assert list(repeat(7, 3)) == [7, 7, 7]
assert tuple(itertools.repeat("a", times=2)) == ("a", "a")
r = repeat(0, -1)
assert list(r) == []
