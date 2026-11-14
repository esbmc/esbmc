arr = [3, 1, 2]
assert sorted(arr) == [1, 2, 3]
assert sorted(arr, reverse=True) == [3, 2, 1]
assert sorted(arr) != [3, 1, 2]
assert arr[0] == 3
