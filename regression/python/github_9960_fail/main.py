# github #9960: `in` on a list of lists compares lists by value.
d = [[1.5], [2.5]]
assert [3.5] not in d
assert [2.5] not in d
