# Test 1: Basic positive integers
q1, r1 = divmod(17, 5)
assert q1 == 3 and r1 == 2

# Test 2: Negative dividend
q2, r2 = divmod(-17, 5)
assert q2 == -4 and r2 == 3

# Test 3: Negative divisor
q3, r3 = divmod(17, -5)
assert q3 == -4 and r3 == -3

# Test 4: Both negative
q4, r4 = divmod(-17, -5)
assert q4 == 3 and r4 == -2

# Test 5: Exact division
q5, r5 = divmod(20, 5)
assert q5 == 4 and r5 == 0

# Test 6: Float division
q6, r6 = divmod(7.5, 2.0)
assert q6 == 3.0 and r6 == 1.5

# Test 7: Mixed types
q7, r7 = divmod(10, 3.0)
assert q7 == 3.0 and r7 == 1.0

# Test 8: Verify divmod property for all cases
assert 17 == q1 * 5 + r1
assert -17 == q2 * 5 + r2
assert 17 == q3 * (-5) + r3
assert -17 == q4 * (-5) + r4
