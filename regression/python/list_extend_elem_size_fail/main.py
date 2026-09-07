# The uniform-scalar-width arm, which is the one the negative pin needs; the
# remaining arms are exercised by the passing twin, and carrying them here only
# made the test unwind past the 120s cap on the macOS runner.
ints = [1, 2]
ints.extend([3, 4])
assert ints[3] == 99
