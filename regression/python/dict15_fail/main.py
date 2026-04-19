d: dict[str, list[int]] = {'a': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]}
l: list[int] = d['a']
assert len(l) == 9
