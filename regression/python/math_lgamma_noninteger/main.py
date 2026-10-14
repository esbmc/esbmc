import math

# Non-integer argument exercises the Lanczos branch of math.lgamma, which adds
# 0.5 * log(2 * pi). With the correct value of pi, lgamma(0.5) equals
# log(sqrt(pi)) = 0.5 * log(pi) = 0.5723649429247001. The former pi = 3.14153
# shifts the result, so the 1e-6 tolerance pins the fix.
r: float = math.lgamma(0.5)
assert math.fabs(r - 0.5723649429247001) < 1e-6
