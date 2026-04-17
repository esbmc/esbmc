is_even = lambda x: x % 2 == 0
assert is_even(4) is True
assert is_even(5) is False

greater = lambda x, y: x > y
assert greater(5, 2) is True
assert greater(2, 5) is False

between = lambda x: 0 <= x <= 10
assert between(5) is True
assert between(-1) is False
