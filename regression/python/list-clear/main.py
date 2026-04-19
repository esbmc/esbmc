# Test 1: Basic clear operation
l = [1, 2, 3]
assert len(l) == 3
l.clear()
assert len(l) == 0

# Test 2: Clear empty list (should not error)
l = []
l.clear()
assert len(l) == 0

# Test 3: Clear and re-populate
l = [1, 2, 3]
l.clear()
assert len(l) == 0
l.append(4)
assert len(l) == 1
assert l[0] == 4

# Test 4: Clear list with mixed types
l = [1, "hello", 3.14, True]
assert len(l) == 4
l.clear()
assert len(l) == 0

# Test 5: Multiple clears (idempotent)
l = [1, 2, 3]
l.clear()
assert len(l) == 0
l.clear()
assert len(l) == 0

# Test 6: Clear doesn't affect other lists
l1 = [1, 2, 3]
l2 = [4, 5, 6]
l1.clear()
assert len(l1) == 0
assert len(l2) == 3
assert l2[0] == 4

# Test 7: Clear after extend
l1 = [1, 2]
l2 = [3, 4]
l1.extend(l2)
assert len(l1) == 4
l1.clear()
assert len(l1) == 0
assert len(l2) == 2  # l2 unchanged
