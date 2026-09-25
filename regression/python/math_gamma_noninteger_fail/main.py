import math

# gamma(0.5) = sqrt(pi) = 1.7724538509055159, which is strictly greater than
# 1.7724. The upper-bound claim below is therefore falsifiable and must yield
# VERIFICATION FAILED — a negative counterpart to math_gamma_noninteger.
r: float = math.gamma(0.5)
assert r < 1.7724
