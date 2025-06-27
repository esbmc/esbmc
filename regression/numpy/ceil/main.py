import numpy as np

# Original test cases
assert np.ceil(0.0) == 0.0
assert np.ceil(1.2) == 2.0
assert np.ceil(-1.8) == -1.0

# Edge Case 1: Integer inputs (should return float equivalents)
assert np.ceil(5) == 5.0
assert np.ceil(-5) == -5.0
assert np.ceil(0) == 0.0

# Edge Case 2: Numbers already at integer values
assert np.ceil(3.0) == 3.0
assert np.ceil(-4.0) == -4.0

# Edge Case 3: Very small positive numbers
assert np.ceil(0.1) == 1.0
assert np.ceil(0.01) == 1.0
assert np.ceil(0.001) == 1.0

# Edge Case 4: Very small negative numbers (should round toward zero)
assert np.ceil(-0.1) == 0.0
assert np.ceil(-0.01) == 0.0
assert np.ceil(-0.001) == 0.0

# Edge Case 5: Numbers very close to integers
assert np.ceil(1.999999) == 2.0
assert np.ceil(1.000001) == 2.0
assert np.ceil(-1.999999) == -1.0
assert np.ceil(-1.000001) == -1.0

# Edge Case 6: Large numbers
assert np.ceil(1000000.5) == 1000001.0
assert np.ceil(-1000000.5) == -1000000.0

# Edge Case 7: Positive and negative zero
assert np.ceil(0.0) == 0.0
assert np.ceil(-0.0) == 0.0

# Edge Case 8: Numbers at floating point precision limits
assert np.ceil(1e-15) == 1.0  # Very small positive
assert np.ceil(-1e-15) == 0.0  # Very small negative

# Edge Case 9: Numbers with many decimal places
assert np.ceil(2.99999999999) == 3.0
assert np.ceil(-2.99999999999) == -2.0

# Edge Case 10: Fractions that should round up
assert np.ceil(0.5) == 1.0
assert np.ceil(-0.5) == 0.0
assert np.ceil(1.5) == 2.0
assert np.ceil(-1.5) == -1.0

# Edge Case 11: Numbers very close to zero from both sides
assert np.ceil(1e-100) == 1.0
assert np.ceil(-1e-100) == 0.0

print("All numpy.ceil edge cases passed!")

assert np.ceil(0.0) == 0.0
assert np.ceil(1.2) == 2.0
assert np.ceil(-1.8) == -1.0
