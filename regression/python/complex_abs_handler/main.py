import math

# Classic Pythagorean triple: |3+4j| = 5.
assert abs(complex(3, 4)) == 5.0

# |0+0j| = 0.
assert abs(complex(0, 0)) == 0.0

# Purely real: |5+0j| = 5.
assert abs(complex(5, 0)) == 5.0

# Purely imaginary: |0+7j| = 7.
assert abs(complex(0, 7)) == 7.0

# Negative components: |-3-4j| = 5.
assert abs(complex(-3, -4)) == 5.0

# Abs of complex expression result.
z = complex(1, 2) + complex(2, 2)
assert abs(z) == 5.0

# Abs with inf component.
abs_inf = abs(complex(float("inf"), 1.0))
assert math.isinf(abs_inf)

# Abs with nan component.
abs_nan = abs(complex(float("nan"), 1.0))
assert math.isnan(abs_nan)

# Signed zero: abs should give positive zero.
abs_sz = abs(complex(-0.0, -0.0))
assert abs_sz == 0.0
assert math.copysign(1.0, abs_sz) == 1.0

# Larger triple: 5-12j -> 13.
assert abs(complex(5, -12)) == 13.0
