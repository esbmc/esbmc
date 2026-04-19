d: dict[str, list[int]] = {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
l: list[int] = d['a']
assert l[0] == 1
assert l[1] == 2
assert l[2] == 3
assert l[3] == 4
assert l[4] == 5
assert l[5] == 6
assert l[6] == 7
assert l[7] == 8
assert l[8] == 9
assert l[9] == 10
