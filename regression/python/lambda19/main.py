pairs = [(1, 3), (2, 1), (0, 5)]

sorted_by_second = sorted(pairs, key=lambda x: x[1])
assert sorted_by_second == [(2, 1), (1, 3), (0, 5)]
