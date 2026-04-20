# all() returns False when any element is falsy, so this assertion is wrong
assert all([1, 0, 3]) == True
