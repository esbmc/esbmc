l = [1, 2, 3]
assert all(x > 0 for x in l) == True

l2 = [1, -1, 3]
assert all(x > 0 for x in l2) == False

l3 = [2, 4, 6]
assert all(x % 2 == 0 for x in l3) == True

l4 = [2, 3, 6]
assert all(x % 2 == 0 for x in l4) == False

# With if clause: filter out negative values before checking
l5 = [1, -2, 3]
assert all(x > 0 for x in l5 if x > 0) == True

# Empty iteration: all() on empty is True (vacuous truth)
l6 = [1, 2, 3]
assert all(x > 10 for x in l6 if x > 10) == True
