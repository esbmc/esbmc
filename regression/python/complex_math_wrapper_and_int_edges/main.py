import math


# Dedicated wrapper behavior through math.isnan/isinf path.
z0 = complex(1.0, 0.0)
z1 = z0
z2 = z1
z3 = z2
z_alias = z3
raised = False
try:
    math.isnan(z_alias)  # type: ignore
except TypeError:
    raised = True
assert raised

raised = False
try:
    math.isinf(z_alias)  # type: ignore
except TypeError:
    raised = True
assert raised


# Integer-like path with deep alias chain.
def get_int_like_payload() -> complex:
    return complex(7.0, 0.0)


payload0 = get_int_like_payload()
payload1 = payload0
payload2 = payload1
payload3 = payload2
n_alias = payload3

raised = False
try:
    math.factorial(n_alias)
except TypeError:
    raised = True
assert raised

raised = False
try:
    math.isqrt(n_alias)
except TypeError:
    raised = True
assert raised
