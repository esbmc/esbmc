assert abs(-5) == 5
assert abs(5) == 5
assert abs(-2.5) == 2.5
assert all([1, 2, 3])
assert not all([1, 0, 3])
assert any([0, 0, 1])
assert not any([0, 0, 0])
assert all([])
assert not any([])
assert sum([1, 2, 3]) == 7
assert min(3, 7) == 3
assert max(3, 7) == 7
assert min([4, 1, 9]) == 1
assert max([4, 1, 9]) == 9
assert bool(1)
assert not bool(0)
