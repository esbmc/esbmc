import math

# Basic integer exponent: (1+1j)**2 = 2j.
z1 = complex(1, 1)
w1 = z1 ** 2
assert abs(w1.real - 0.0) < 1e-10
assert abs(w1.imag - 2.0) < 1e-10

# Exponent 0: any complex ** 0 = 1+0j.
z2 = complex(3, 4)
w2 = z2 ** 0
assert w2.real == 1.0
assert w2.imag == 0.0

# Exponent 1: identity.
w3 = z2 ** 1
assert w3 == z2

# Negative integer exponent: z ** -1 = 1/z.
z4 = complex(1, 1)
w4 = z4 ** (-1)
# 1/(1+1j) = (1-1j)/2 = 0.5 - 0.5j
assert abs(w4.real - 0.5) < 1e-10
assert abs(w4.imag - (-0.5)) < 1e-10

# Exponent 3: (1+1j)**3 = (1+1j)*(2j) = -2+2j.
w5 = complex(1, 1) ** 3
assert abs(w5.real - (-2.0)) < 1e-10
assert abs(w5.imag - 2.0) < 1e-10

# Bool exponent: True -> exponent 1, False -> exponent 0.
z6 = complex(5, 12)
assert z6 ** True == z6
w6_false = z6 ** False
assert w6_false.real == 1.0
assert w6_false.imag == 0.0

# TypeError on unsupported types for **.
raised = False
try:
    _ = complex(1, 2) ** "2"  # type: ignore
except TypeError:
    raised = True
assert raised
