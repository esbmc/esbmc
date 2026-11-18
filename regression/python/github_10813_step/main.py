arr = [1, 1.0, 4]
n = 1
assert sum(arr[::2]) == 5
assert sum(arr[-2:]) == 5
assert sum(arr[::-1]) == 6
assert sum(arr[:n]) == 1
