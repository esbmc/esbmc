arr = [1, 1.0]
assert sum(arr[:1]) == 1
assert sum(arr[1:]) == 1.0
assert sum(arr[:1], 2) == 3
