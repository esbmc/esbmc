import math

z = complex(1.0, 2.0)

r_sin = False
try:
    math.sin(z)  # type: ignore
except TypeError:
    r_sin = True
assert r_sin

r_sqrt = False
try:
    math.sqrt(z)  # type: ignore
except TypeError:
    r_sqrt = True
assert r_sqrt

r_atan2 = False
try:
    math.atan2(z, 1.0)  # type: ignore
except TypeError:
    r_atan2 = True
assert r_atan2

r_hypot = False
try:
    math.hypot(z, 1.0)  # type: ignore
except TypeError:
    r_hypot = True
assert r_hypot

r_floor = False
try:
    math.floor(z)  # type: ignore
except TypeError:
    r_floor = True
assert r_floor

r_ceil = False
try:
    math.ceil(z)  # type: ignore
except TypeError:
    r_ceil = True
assert r_ceil

r_isfinite = False
try:
    math.isfinite(z)  # type: ignore
except TypeError:
    r_isfinite = True
assert r_isfinite

r_degrees = False
try:
    math.degrees(z)  # type: ignore
except TypeError:
    r_degrees = True
assert r_degrees

r_ldexp = False
try:
    math.ldexp(z, 2)  # type: ignore
except TypeError:
    r_ldexp = True
assert r_ldexp

r_remainder = False
try:
    math.remainder(z, 2.0)  # type: ignore
except TypeError:
    r_remainder = True
assert r_remainder

r_remainder_unpack = False
kw_remainder = {"x": z, "y": 2.0}
try:
    math.remainder(**kw_remainder)  # type: ignore
except TypeError:
    r_remainder_unpack = True
assert r_remainder_unpack

r_remainder_mixed = False
try:
    math.remainder(1.0, y=z)  # type: ignore
except TypeError:
    r_remainder_mixed = True
assert r_remainder_mixed

r_isclose = False
try:
    math.isclose(a=1.0, b=z)  # type: ignore
except TypeError:
    r_isclose = True
assert r_isclose

r_isclose_unpack = False
kw_isclose = {"a": 1.0, "b": z}
try:
    math.isclose(**kw_isclose)  # type: ignore
except TypeError:
    r_isclose_unpack = True
assert r_isclose_unpack

r_factorial = False
try:
    math.factorial(z)  # type: ignore
except TypeError:
    r_factorial = True
assert r_factorial

r_gcd = False
try:
    math.gcd(z, 1)  # type: ignore
except TypeError:
    r_gcd = True
assert r_gcd

r_lcm = False
try:
    math.lcm(z, 1)  # type: ignore
except TypeError:
    r_lcm = True
assert r_lcm

r_isqrt = False
try:
    math.isqrt(z)  # type: ignore
except TypeError:
    r_isqrt = True
assert r_isqrt

r_perm = False
try:
    math.perm(z, 1)  # type: ignore
except TypeError:
    r_perm = True
assert r_perm

r_fsum = False
try:
    math.fsum([1.0, z])  # type: ignore
except TypeError:
    r_fsum = True
assert r_fsum

r_dist = False
try:
    math.dist((z, 0.0), (0.0, 0.0))  # type: ignore
except TypeError:
    r_dist = True
assert r_dist
