# Verifies only the FIRST occurrence is removed
l = [5, 5, 5]
l.remove(5)
assert l[0] == 5
assert l[1] == 5
