# All True
assert all([True, True, True]) == True

# At least one False
assert all([True, False, True]) == False

# Empty list
assert all([]) == True