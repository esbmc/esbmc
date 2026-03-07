import math

z = complex(3, 4)
assert abs(z) == 5.0

z_expr = complex(1, 2) + complex(2, 1)
assert abs(z_expr) == math.sqrt(18.0)

z0 = complex(0, 0)
z1 = complex(0, 2)

if z0:
    assert False

if z1:
    pass
else:
    assert False

count = 0
z_loop = complex(1, 0)
while z_loop:
    count = count + 1
    z_loop = complex(0, 0)
assert count == 1

t_ifexp = 10 if complex(1, 0) else 20
assert t_ifexp == 10
t_ifexp_zero = 10 if complex(0, 0) else 20
assert t_ifexp_zero == 20

assert ((1 + 0j) == 1)
assert ((1 + 0j) != 2)
assert ((1 + 0j) != "1")  # type: ignore[comparison-overlap]

assert complex(1, 2).conjugate() == complex(1, -2)
assert complex(1, 2).conjugate().conjugate() == complex(1, 2)

assert bool(complex(float("nan"), 0.0)) == True
assert bool(complex(float("inf"), 0.0)) == True
assert bool(complex(-0.0, 0.0)) == False
assert bool(complex(0.0, -0.0)) == False

z_inf_src = complex(float("inf"), float("-inf"))
z_conj_inf = z_inf_src.conjugate()
assert z_conj_inf.real == float("inf")
assert z_conj_inf.imag == float("inf")

z_nan_src = complex(float("nan"), 1.0)
z_conj_nan = z_nan_src.conjugate()
assert z_conj_nan != z_conj_nan

abs_nan = abs(complex(float("nan"), 1.0))
assert math.isnan(abs_nan)
abs_inf = abs(complex(float("inf"), 1.0))
assert math.isinf(abs_inf)

z_nan_eq = complex(float("nan"), 0.0)
assert z_nan_eq != z_nan_eq
assert not (z_nan_eq == z_nan_eq)

# Edge 1: abs() with signed-zero components keeps numeric zero.
abs_signed_zero = abs(complex(-0.0, -0.0))
assert abs_signed_zero == 0.0
assert math.copysign(1.0, abs_signed_zero) == 1.0

# Edge 2: conjugate() preserves real signed zero and flips imag signed zero.
z_signed_zero = complex(-0.0, -0.0)
z_signed_zero_conj = z_signed_zero.conjugate()
assert math.copysign(1.0, z_signed_zero_conj.real) == -1.0
assert math.copysign(1.0, z_signed_zero_conj.imag) == 1.0

# Edge 3: equality with inf/-inf in components.
assert complex(float("inf"), 1.0) == complex(float("inf"), 1.0)
assert complex(float("inf"), 1.0) != complex(float("-inf"), 1.0)

# Edge 4: inequality with mixed inf sign on imag component.
assert complex(1.0, float("inf")) != complex(1.0, float("-inf"))

# Edge 5: bool() with subnormal real part is True.
assert bool(complex(5e-324, 0.0)) == True

# Edge 6: bool() with subnormal imag part is True.
assert bool(complex(0.0, -5e-324)) == True

raised = False
try:
    _ = (1 + 2j) < (3 + 4j)  # type: ignore[operator]
except TypeError:
    raised = True
assert raised

raised = False
try:
    _ = (1 + 2j) < 3  # type: ignore[operator]
except TypeError:
    raised = True
assert raised

raised = False
try:
    _ = 3 < (1 + 2j)  # type: ignore[operator]
except TypeError:
    raised = True
assert raised

raised = False
try:
    _ = (1 + 2j) // 2  # type: ignore[operator]
except TypeError:
    raised = True
assert raised

raised = False
try:
    _ = (1 + 2j) // 2.0  # type: ignore[operator]
except TypeError:
    raised = True
assert raised

raised = False
try:
    _ = (1 + 2j) // True  # type: ignore[operator]
except TypeError:
    raised = True
assert raised

raised = False
try:
    _ = (1 + 2j) % 2  # type: ignore[operator]
except TypeError:
    raised = True
assert raised

raised = False
try:
    _ = (1 + 2j) % 2.0  # type: ignore[operator]
except TypeError:
    raised = True
assert raised

raised = False
try:
    _ = (1 + 2j) % True  # type: ignore[operator]
except TypeError:
    raised = True
assert raised
