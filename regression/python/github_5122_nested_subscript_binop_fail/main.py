# Non-vacuity counterpart of github_5122_nested_subscript_binop: the nested
# constant double-subscript read returns the genuine value (3.0 + 4.0 == 7.0),
# so asserting a wrong value must FAIL with a counterexample -- proving the
# fix resolves the real element type rather than vacuously passing (#5122).

M = [[3.0, 4.0]]
d = M[0][0] + M[0][1]
assert d == 99.0
