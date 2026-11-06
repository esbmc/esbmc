# github #9960: `in` on a list of lists compares lists by value.
def f() -> list[int]:
    return [4]


d = [[1.5], [2.5]]
assert [1.5] in d
assert [2.5] in d
assert [3.5] not in d
assert [1] in [[1.0], [2]]
assert [[1], [2]] in [[[0]], [[1], [2]]]
assert [1] in [None, [1]]
assert f() in [[4]]
assert [9007199254740993] not in [[9007199254740992.0]]
assert [(1, 2)] not in [[(1, 3)]]
