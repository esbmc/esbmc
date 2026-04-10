lst: list[int] = [10, 20, 30]
it = lst.__iter__()
total: int = 0
for x in it:
    total = total + x
assert total == 60
