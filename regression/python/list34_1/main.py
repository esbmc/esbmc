lst = [4, 5]
r = lst * 2
assert r == [4, 5, 4, 5]

r2 = 2 * lst
assert r2 == [4, 5, 4, 5]

empty: list[int] = []
assert empty * 3 == []

n = 2
lst2 = [1, 2]
assert lst2 * n == [1, 2, 1, 2]
