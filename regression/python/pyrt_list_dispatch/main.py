items = [1, 2]
items.append(3)
assert len(items) == 3
assert items[-1] == 3
items[0] = 7
assert items[0] + items[1] == 9
