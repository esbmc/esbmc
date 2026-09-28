# list.sort() dispatches on the frontend's type_flag, so every element family
# must still sort by its own ordering, not by a byte comparison.
ints = [7, 3, -2, 0, 3]
ints.sort()
assert ints == [-2, 0, 3, 3, 7]

floats = [1.5, -0.5, 2.25]
floats.sort()
assert floats == [-0.5, 1.5, 2.25]

bools = [True, False, True]
bools.sort()
assert bools == [False, True, True]

mixed = [3, 1.5, -2]
mixed.sort()
assert mixed[0] == -2
assert mixed[2] == 3

words = ["ab", "", "a"]
words.sort()
assert words == ["", "a", "ab"]
