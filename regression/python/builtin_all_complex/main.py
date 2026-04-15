assert all([1 + 2j, 3j, 0.5 + 0j]) == True
assert all([1 + 0j, 2 + 0j]) == True
assert all([0j, 1]) == False
assert all([1, 0 + 0j]) == False
