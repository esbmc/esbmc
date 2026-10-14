import math

# Non-integer argument exercises the Lanczos branch of math.gamma, which
# scales by sqrt(2 * pi). With the correct value of pi, gamma(0.5) equals
# sqrt(pi) = 1.7724538509055159 to full working precision. A wrong constant
# (the former pi = 3.14153) shifts the result by ~1.8e-5, so the 1e-6
# tolerance below pins the fix.
r: float = math.gamma(0.5)
assert math.fabs(r - 1.7724538509055159) < 1e-6
