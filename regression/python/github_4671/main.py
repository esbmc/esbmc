import cmath

r, phi = cmath.polar(complex(0.0, 0.0))
assert r == 0.0

z = cmath.rect(1.0, 0.0)
assert z.real == 1.0
