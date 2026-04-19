single = b'A'
assert len(single) == 1
assert single[0] == 65  # ASCII value of 'A'
assert single[-1] == 65  # Negative indexing works, -1 refers to the last element (same as index 0 here)
