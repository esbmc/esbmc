l = [3, 2, 1]
result = sorted(l)
assert result[0] == 1  # Check the sorted result
assert result[1] == 2
assert result[2] == 3
assert l[0] == 3  # Original list unchanged
assert l[1] == 2
assert l[2] == 1
