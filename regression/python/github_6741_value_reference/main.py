def count(*pos):
    return len(pos)


take = count
assert count(1, 2) == 2
