import numpy as np

# Unary-negated float matrix elements exercise the UnaryOp branch of the
# numeric pre-check in try_extract_numeric_constant (issue #5206). Verify a
# negative-float matrix is still accepted and the determinant is correct:
# det([[-2.0, 1.0], [4.0, -3.0]]) = (-2.0)(-3.0) - (1.0)(4.0) = 2.0
m = np.array([[-2.0, 1.0], [4.0, -3.0]])
x = np.linalg.det(m)
assert x == 2.0
