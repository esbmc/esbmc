# Negative: the comprehension really populates the new dict, so a wrong
# expectation must FAIL (guards against the desugaring becoming a no-op).
d = {('A', 'B'): 3, ('C', 'D'): 7}
m = {v: 1 for u, v in d}
assert m['B'] == 0
