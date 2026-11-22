arr = [3, 1.0]
assert min(arr[:1]) == 3
assert max(arr[:1]) == 3
assert max(arr[1:]) == 1.0
