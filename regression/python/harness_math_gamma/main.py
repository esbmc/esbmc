# Verification harness for math.gamma (src/python-frontend/models/math.py).
#
# For a positive integer x, gamma(x) == (x - 1)!.  The model computes this
# branch exactly (an integer factorial widened to float), so the harness uses
# concrete integer anchors: the general real-valued branch is a transcendental
# Lanczos approximation whose float encoding is not amenable to symbolic
# exploration here.
#
# ENSURES:
#   E1: gamma(n) == (n - 1)! for n = 1..5        [integer-argument contract]
import math

assert math.gamma(1.0) == 1.0      # 0! == 1
assert math.gamma(2.0) == 1.0      # 1! == 1
assert math.gamma(3.0) == 2.0      # 2! == 2
assert math.gamma(4.0) == 6.0      # 3! == 6
assert math.gamma(5.0) == 24.0     # 4! == 24
