import math

assert (3 + 4j) == complex(3, 4)
assert ((1 + 2j) + 3) == complex(4, 2)
assert (3 + (1 + 2j)) == complex(4, 2)
assert ((1 + 2j) * (3 + 4j)) == complex(-5, 10)
assert ((1 + 2j) - (3 + 4j)) == complex(-2, -2)
assert ((1 + 2j) / (1 + 1j)) == complex(1.5, 0.5)

assert ((1 + 2j) - 3) == complex(-2, 2)
assert (3 - (1 + 2j)) == complex(2, -2)
assert ((4 + 2j) / 2) == complex(2, 1)
assert (2 / (1 - 1j)) == complex(1, 1)
assert ((1 + 2j) + True) == complex(2, 2)
assert (True + (1 + 2j)) == complex(2, 2)
assert ((1 + 2j) - True) == complex(0, 2)
assert (True - (1 + 2j)) == complex(0, -2)
assert ((1 + 2j) * True) == complex(1, 2)
assert (True * (1 + 2j)) == complex(1, 2)
assert ((1 + 2j) / True) == complex(1, 2)

assert ((1 + 0j) == 1)
assert ((1 + 2j) != (1 + 3j))

# Edge: comparison against non-numeric must follow Python semantics.
assert not ((1 + 0j) == "1")  # type: ignore[comparison-overlap]
assert ((1 + 0j) != "1")  # type: ignore[comparison-overlap]

# Edge: mixed float + complex promotion
assert (2.5 + (1 + 2j)) == complex(3.5, 2.0)

# Edge: UnaryOp over complex constant should preserve sign in literal conversion.
assert (-1j) == complex(0, -1)

# Phase 4: unary operators on complex expressions/variables.
z = complex(1, 2)
assert (-z) == complex(-1, -2)
assert (+z) == complex(1, 2)
assert (-(-z)) == complex(1, 2)
assert (-(+z)) == complex(-1, -2)
assert (+(+z)) == complex(1, 2)

# Edge: precedence and grouped expressions.
assert ((1 + 2j) * (3 + 4j) - 2) == complex(-7, 10)
assert ((1 + 2j) / (2 + 2j)) == complex(0.75, 0.25)
assert (1 / (2 + 2j)) == complex(0.25, -0.25)

# Phase 5 basic builtins/methods compatibility for complex.
assert abs(complex(3, 4)) == 5.0
assert bool(complex(0, 0)) == False
assert bool(complex(0, 2)) == True
assert complex(1, 2).conjugate() == complex(1, -2)

# Edge: invalid operations must raise TypeError.
raised = False
try:
    _ = (1 + 2j) < (3 + 4j)  # type: ignore[operator]
except TypeError:
    raised = True
assert raised

# Edge: inf/nan propagation from float parts.
z_inf = complex(float("inf"), -float("inf"))
assert z_inf.real == float("inf")
assert z_inf.imag == float("-inf")

z_nan_r = complex(float("nan"), 0.0)
z_nan_i = complex(0.0, float("nan"))
assert z_nan_r != z_nan_r
assert z_nan_i != z_nan_i

# Edge: division by zero compatibility:
# CPython raises, ESBMC currently may model IEEE result.
zero_div_raised = False
z_div0 = complex(0.0, 0.0)
try:
    z_div0 = (1 + 2j) / (0 + 0j)
except ZeroDivisionError:
    zero_div_raised = True

if not zero_div_raised:
    assert (z_div0 != z_div0 or z_div0.real == float("inf") or z_div0.real == float("-inf")
            or z_div0.imag == float("inf") or z_div0.imag == float("-inf"))

# Edge: sign of zero in components.
z_neg_zero = complex(-0.0, 0.0)
assert z_neg_zero == complex(0.0, 0.0)
assert math.copysign(1.0, z_neg_zero.real) == -1.0
assert math.copysign(1.0, z_neg_zero.imag) == 1.0

# Edge: sign propagation of -0.0 through unary operation.
z_zero_signed = complex(0.0, -0.0)
z_zero_signed_neg = -z_zero_signed
assert z_zero_signed_neg == complex(-0.0, 0.0)
assert z_zero_signed_neg.real == 0.0
assert z_zero_signed_neg.imag == 0.0

# Edge: strict signed-zero checks in +, -, *, /.
z_s0_a = complex(-0.0, 0.0)
z_s0_b = complex(-0.0, 0.0)
z_s0_add = z_s0_a + z_s0_b
assert math.copysign(1.0, z_s0_add.real) == -1.0
assert math.copysign(1.0, z_s0_add.imag) == 1.0

z_s0_sub = z_s0_a - complex(0.0, -0.0)
assert math.copysign(1.0, z_s0_sub.real) == -1.0
assert math.copysign(1.0, z_s0_sub.imag) == 1.0

z_s0_mul = z_s0_a * complex(1.0, 0.0)
assert math.copysign(1.0, z_s0_mul.real) == -1.0
assert math.copysign(1.0, z_s0_mul.imag) == 1.0

z_s0_div = z_s0_a / complex(1.0, 0.0)
assert math.copysign(1.0, z_s0_div.real) == 1.0
assert math.copysign(1.0, z_s0_div.imag) == 1.0

# Edge: overflow/underflow behavior in mixed arithmetic.
z_over = complex(1e308, 1e308) * complex(2.0, 0.0)
assert (z_over != z_over or z_over.real == float("inf") or z_over.real == float("-inf")
        or z_over.imag == float("inf") or z_over.imag == float("-inf"))

z_under = complex(1e-308, -1e-308) / complex(2.0, 0.0)
assert z_under.real != 0.0 or z_under.imag != 0.0

# Edge: more exotic nan/inf combinations.
z_nan_mix_add = complex(float("inf"), 1.0) + complex(float("-inf"), 2.0)
assert math.isnan(z_nan_mix_add.real)
assert z_nan_mix_add.imag == 3.0

z_nan_mix_sub = complex(float("inf"), 1.0) - complex(float("inf"), 2.0)
assert math.isnan(z_nan_mix_sub.real)
assert z_nan_mix_sub.imag == -1.0

z_inf_mix_mul = complex(float("inf"), 1.0) * complex(2.0, 0.0)
assert math.isinf(z_inf_mix_mul.real)
assert math.isnan(z_inf_mix_mul.imag)

z_small_div_inf = complex(1.0, -1.0) / complex(float("inf"), 0.0)
assert (z_small_div_inf.real == 0.0 or math.isnan(z_small_div_inf.real)
        or math.isinf(z_small_div_inf.real))
assert (z_small_div_inf.imag == 0.0 or math.isnan(z_small_div_inf.imag)
        or math.isinf(z_small_div_inf.imag))

# Edge: comparison-oriented checks for special values.
z_cmp_nan = complex(float("nan"), 1.0)
assert z_cmp_nan != z_cmp_nan
z_cmp_inf = complex(float("inf"), 0.0)
assert z_cmp_inf == z_cmp_inf

# Edge: longer precedence/associativity chain.
a = complex(2.0, 3.0)
b = complex(4.0, -1.0)
c = 2.0
d = 3.0
e = complex(1.0, -2.0)
lhs = ((a + b) / c) * d - e
rhs = (((a + b) / c) * d) - e
assert lhs == rhs

raised = False
try:
    _ = (1 + 2j) // 2  # type: ignore[operator]
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
    _ = (1 + 2j) + "x"  # type: ignore[operator]
except TypeError:
    raised = True
assert raised
