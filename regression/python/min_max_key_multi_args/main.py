assert max(3, 1, 2, key=lambda x: -x) == 1
assert min(3, 1, 2, key=lambda x: -x) == 3
assert max(-4, 3, key=abs) == -4
assert max(2, 5, 5, key=lambda x: x) == 5
