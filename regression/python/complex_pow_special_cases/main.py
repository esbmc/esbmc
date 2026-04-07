# Tests i**4 = 1 (power by squaring loop)
# and various exponent special cases

# i**4 = (0+1j)**4 = 1+0j  (cycle: i -> -1 -> -i -> 1)
w = complex(0, 1) ** 4
assert w.real > 0.99 and w.real < 1.01
assert w.imag > -0.01 and w.imag < 0.01

# (1+0j)**10 = 1+0j (identity base)
v = complex(1, 0) ** 10
assert v.real > 0.99 and v.real < 1.01
assert v.imag > -0.01 and v.imag < 0.01

# z**0 = 1+0j for any z
z = complex(99, -42)
r = z ** 0
assert r.real > 0.99 and r.real < 1.01
assert r.imag > -0.01 and r.imag < 0.01

# Bool exponent True = 1: z**True == z
z2 = complex(3, 4)
b = z2 ** True
assert b == complex(3, 4)

# Bool exponent False = 0: z**False == 1+0j
b2 = z2 ** False
assert b2.real > 0.99 and b2.real < 1.01
assert b2.imag > -0.01 and b2.imag < 0.01
