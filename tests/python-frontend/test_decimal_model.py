"""Property-based tests validating the Decimal model against CPython's decimal.Decimal."""

import os
import importlib.util
import decimal as cpython_decimal

import pytest
from hypothesis import given, strategies as st, assume, settings, HealthCheck


def load_model_decimal():
    """Load the Decimal class from the ESBMC operational model."""
    model_path = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "src", "python-frontend", "models", "decimal.py"
    )
    model_path = os.path.abspath(model_path)
    spec = importlib.util.spec_from_file_location("decimal_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.Decimal


ModelDecimal = load_model_decimal()


def cpython_to_model(d):
    """Convert a CPython Decimal to our model's 4-field representation."""
    t = d.as_tuple()
    sign = t.sign
    if t.exponent == 'n':
        return sign, 0, 0, 2
    elif t.exponent == 'N':
        return sign, 0, 0, 3
    elif t.exponent == 'F':
        return sign, 0, 0, 1
    else:
        int_val = 0
        for digit in t.digits:
            int_val = int_val * 10 + digit
        return sign, int_val, t.exponent, 0


# --- Strategies ---

finite_decimals = st.builds(
    lambda sign, digits, exp: cpython_decimal.Decimal(
        (sign, tuple(digits), exp)
    ),
    sign=st.integers(min_value=0, max_value=1),
    digits=st.lists(st.integers(min_value=0, max_value=9), min_size=1, max_size=10),
    exp=st.integers(min_value=-20, max_value=20),
)

special_decimals = st.sampled_from([
    cpython_decimal.Decimal('NaN'),
    cpython_decimal.Decimal('-NaN'),
    cpython_decimal.Decimal('sNaN'),
    cpython_decimal.Decimal('-sNaN'),
    cpython_decimal.Decimal('Infinity'),
    cpython_decimal.Decimal('-Infinity'),
])

zero_decimals = st.sampled_from([
    cpython_decimal.Decimal('0'),
    cpython_decimal.Decimal('-0'),
    cpython_decimal.Decimal('0.0'),
    cpython_decimal.Decimal('-0.00'),
])

all_decimals = st.one_of(finite_decimals, special_decimals, zero_decimals)


def values_equal(m, expected_tuple):
    """Compare model Decimal against a cpython_to_model tuple by value.

    Handles: NaN identity, zero sign ambiguity, exponent normalization.
    """
    es, ei, ee, esp = expected_tuple
    if m._is_special != esp:
        return False
    if m._is_special >= 2:
        return True
    if m._is_special == 1:
        return m._sign == es
    if m._int == 0 and ei == 0:
        return True
    exp = ModelDecimal(es, ei, ee, esp)
    return m.__eq__(exp) and m._sign == es


# --- Construction Tests ---

class TestConstruction:
    @given(d=finite_decimals)
    @settings(max_examples=200)
    def test_finite_round_trip(self, d):
        """Model construction preserves CPython's internal representation."""
        sign, int_val, exp, is_special = cpython_to_model(d)
        m = ModelDecimal(sign, int_val, exp, is_special)
        assert m._sign == sign
        assert m._int == int_val
        assert m._exp == exp
        assert m._is_special == is_special

    @given(d=special_decimals)
    def test_special_values(self, d):
        """Special values (NaN, sNaN, Inf) are represented correctly."""
        sign, int_val, exp, is_special = cpython_to_model(d)
        m = ModelDecimal(sign, int_val, exp, is_special)
        assert m._sign == sign
        assert m._is_special == is_special
        if d.is_nan():
            assert is_special in (2, 3)
        if d.is_infinite():
            assert is_special == 1

    @given(d=zero_decimals)
    def test_zero_variants(self, d):
        """Zeros with different signs and exponents are preserved."""
        sign, int_val, exp, is_special = cpython_to_model(d)
        m = ModelDecimal(sign, int_val, exp, is_special)
        assert m._sign == sign
        assert m._int == 0
        assert m._exp == exp
        assert m._is_special == 0


class TestDecimalFromInt:
    @given(n=st.integers(min_value=-10000, max_value=10000))
    @settings(max_examples=200)
    def test_from_int_matches_cpython(self, n):
        """Decimal._from_int produces same representation as CPython Decimal(n)."""
        m = ModelDecimal._from_int(n)
        d = cpython_decimal.Decimal(n)
        sign, int_val, exp, is_special = cpython_to_model(d)
        assert m._sign == sign
        assert m._int == int_val
        assert m._exp == exp
        assert m._is_special == is_special


class TestPreprocessorRewriting:
    """Test that the preprocessor rewriting logic produces correct values.

    These tests simulate what the preprocessor does: take a user-facing
    Decimal(...) call, evaluate it with CPython, extract fields, and
    verify the model constructor receives the right values.
    """

    @pytest.mark.parametrize("input_val,expected_sign,expected_int,expected_exp,expected_special", [
        ("0", 0, 0, 0, 0),
        ("-0", 1, 0, 0, 0),
        ("1", 0, 1, 0, 0),
        ("-1", 1, 1, 0, 0),
        ("3.14", 0, 314, -2, 0),
        ("-3.14", 1, 314, -2, 0),
        ("10.5", 0, 105, -1, 0),
        ("100", 0, 100, 0, 0),
        ("0.001", 0, 1, -3, 0),
        ("NaN", 0, 0, 0, 2),
        ("-NaN", 1, 0, 0, 2),
        ("sNaN", 0, 0, 0, 3),
        ("Infinity", 0, 0, 0, 1),
        ("-Infinity", 1, 0, 0, 1),
    ])
    def test_string_construction(self, input_val, expected_sign, expected_int, expected_exp, expected_special):
        d = cpython_decimal.Decimal(input_val)
        sign, int_val, exp, is_special = cpython_to_model(d)
        assert sign == expected_sign
        assert int_val == expected_int
        assert exp == expected_exp
        assert is_special == expected_special

    @pytest.mark.parametrize("input_val", [0, 1, -1, 42, -42, 1000000])
    def test_int_construction(self, input_val):
        d = cpython_decimal.Decimal(input_val)
        sign, int_val, exp, is_special = cpython_to_model(d)
        m = ModelDecimal(sign, int_val, exp, is_special)
        assert m._is_special == 0
        assert m._exp == 0

    def test_float_construction(self):
        d = cpython_decimal.Decimal(3.14)
        sign, int_val, exp, is_special = cpython_to_model(d)
        m = ModelDecimal(sign, int_val, exp, is_special)
        assert m._is_special == 0
        assert m._sign == 0

    def test_empty_construction(self):
        d = cpython_decimal.Decimal()
        sign, int_val, exp, is_special = cpython_to_model(d)
        assert sign == 0
        assert int_val == 0
        assert exp == 0
        assert is_special == 0


class TestEquality:
    @given(d=finite_decimals)
    @settings(max_examples=200)
    def test_equality_reflexive(self, d):
        """Every finite Decimal equals itself."""
        s, i, e, sp = cpython_to_model(d)
        m = ModelDecimal(s, i, e, sp)
        assert m.__eq__(m) == True

    @given(a=finite_decimals, b=finite_decimals)
    @settings(max_examples=200)
    def test_equality_matches_cpython(self, a, b):
        """Model __eq__ matches CPython for finite decimals."""
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        assert ma.__eq__(mb) == (a == b)

    @given(d=all_decimals)
    @settings(max_examples=200)
    def test_ne_is_negation_of_eq(self, d):
        """__ne__ is always the negation of __eq__."""
        s, i, e, sp = cpython_to_model(d)
        m1 = ModelDecimal(s, i, e, sp)
        m2 = ModelDecimal(s, i, e, sp)
        assert m1.__ne__(m2) == (not m1.__eq__(m2))

    def test_nan_not_equal_to_self(self):
        assert ModelDecimal(0, 0, 0, 2).__eq__(ModelDecimal(0, 0, 0, 2)) == False

    def test_snan_not_equal_to_self(self):
        assert ModelDecimal(0, 0, 0, 3).__eq__(ModelDecimal(0, 0, 0, 3)) == False

    def test_neg_zero_equals_pos_zero(self):
        assert ModelDecimal(1, 0, 0, 0).__eq__(ModelDecimal(0, 0, 0, 0)) == True

    def test_different_exponents_same_value(self):
        """1.0 == 1.00"""
        assert ModelDecimal(0, 10, -1, 0).__eq__(ModelDecimal(0, 100, -2, 0)) == True

    def test_infinity_equals_same_sign(self):
        assert ModelDecimal(0, 0, 0, 1).__eq__(ModelDecimal(0, 0, 0, 1)) == True
        assert ModelDecimal(1, 0, 0, 1).__eq__(ModelDecimal(1, 0, 0, 1)) == True

    def test_pos_inf_not_equal_neg_inf(self):
        assert ModelDecimal(0, 0, 0, 1).__eq__(ModelDecimal(1, 0, 0, 1)) == False


class TestOrdering:
    @given(a=finite_decimals, b=finite_decimals)
    @settings(max_examples=200)
    def test_lt_matches_cpython(self, a, b):
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        assert ma.__lt__(mb) == (a < b)

    @given(a=finite_decimals, b=finite_decimals)
    @settings(max_examples=200)
    def test_le_matches_cpython(self, a, b):
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        assert ma.__le__(mb) == (a <= b)

    @given(a=finite_decimals, b=finite_decimals)
    @settings(max_examples=200)
    def test_gt_matches_cpython(self, a, b):
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        assert ma.__gt__(mb) == (a > b)

    @given(a=finite_decimals, b=finite_decimals)
    @settings(max_examples=200)
    def test_ge_matches_cpython(self, a, b):
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        assert ma.__ge__(mb) == (a >= b)

    @given(d=finite_decimals)
    @settings(max_examples=200)
    def test_lt_nan_always_false(self, d):
        s, i, e, sp = cpython_to_model(d)
        m = ModelDecimal(s, i, e, sp)
        nan = ModelDecimal(0, 0, 0, 2)
        assert m.__lt__(nan) == False
        assert nan.__lt__(m) == False

    @given(d=finite_decimals)
    @settings(max_examples=200)
    def test_gt_nan_always_false(self, d):
        s, i, e, sp = cpython_to_model(d)
        m = ModelDecimal(s, i, e, sp)
        nan = ModelDecimal(0, 0, 0, 2)
        assert m.__gt__(nan) == False
        assert nan.__gt__(m) == False

    @given(d=finite_decimals)
    @settings(max_examples=200)
    def test_le_nan_always_false(self, d):
        s, i, e, sp = cpython_to_model(d)
        m = ModelDecimal(s, i, e, sp)
        nan = ModelDecimal(0, 0, 0, 2)
        assert m.__le__(nan) == False
        assert nan.__le__(m) == False

    @given(d=finite_decimals)
    @settings(max_examples=200)
    def test_ge_nan_always_false(self, d):
        s, i, e, sp = cpython_to_model(d)
        m = ModelDecimal(s, i, e, sp)
        nan = ModelDecimal(0, 0, 0, 2)
        assert m.__ge__(nan) == False
        assert nan.__ge__(m) == False

    def test_snan_ordering_always_false(self):
        snan = ModelDecimal(0, 0, 0, 3)
        val = ModelDecimal(0, 1, 0, 0)
        assert snan.__lt__(val) == False
        assert snan.__le__(val) == False
        assert snan.__gt__(val) == False
        assert snan.__ge__(val) == False
        assert val.__lt__(snan) == False
        assert val.__le__(snan) == False
        assert val.__gt__(snan) == False
        assert val.__ge__(snan) == False

    def test_infinity_ordering(self):
        pos_inf = ModelDecimal(0, 0, 0, 1)
        neg_inf = ModelDecimal(1, 0, 0, 1)
        finite = ModelDecimal(0, 42, 0, 0)
        assert pos_inf.__gt__(finite) == True
        assert pos_inf.__ge__(finite) == True
        assert pos_inf.__lt__(finite) == False
        assert pos_inf.__le__(finite) == False
        assert neg_inf.__lt__(finite) == True
        assert neg_inf.__le__(finite) == True
        assert neg_inf.__gt__(finite) == False
        assert neg_inf.__ge__(finite) == False
        assert neg_inf.__lt__(pos_inf) == True
        assert pos_inf.__gt__(neg_inf) == True
        assert pos_inf.__lt__(pos_inf) == False
        assert pos_inf.__le__(pos_inf) == True
        assert pos_inf.__ge__(pos_inf) == True

    def test_neg_zero_ordering(self):
        neg_zero = ModelDecimal(1, 0, 0, 0)
        pos_zero = ModelDecimal(0, 0, 0, 0)
        assert neg_zero.__lt__(pos_zero) == False
        assert neg_zero.__gt__(pos_zero) == False
        assert neg_zero.__le__(pos_zero) == True
        assert neg_zero.__ge__(pos_zero) == True

    @given(a=finite_decimals, b=finite_decimals)
    @settings(max_examples=200)
    def test_ordering_consistency(self, a, b):
        """a < b iff b > a, a <= b iff b >= a."""
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        assert ma.__lt__(mb) == mb.__gt__(ma)
        assert ma.__le__(mb) == mb.__ge__(ma)

    @given(a=finite_decimals, b=finite_decimals)
    @settings(max_examples=200)
    def test_le_is_lt_or_eq(self, a, b):
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        assert ma.__le__(mb) == (ma.__lt__(mb) or ma.__eq__(mb))

    @given(a=finite_decimals, b=finite_decimals)
    @settings(max_examples=200)
    def test_ge_is_gt_or_eq(self, a, b):
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        assert ma.__ge__(mb) == (ma.__gt__(mb) or ma.__eq__(mb))


nonzero_finite_decimals = finite_decimals.filter(lambda d: d != 0)

# Narrower strategy for division/mod to stay within CPython's 28-digit precision
small_finite_decimals = st.builds(
    lambda sign, digits, exp: cpython_decimal.Decimal(
        (sign, tuple(digits), exp)
    ),
    sign=st.integers(min_value=0, max_value=1),
    digits=st.lists(st.integers(min_value=0, max_value=9), min_size=1, max_size=6),
    exp=st.integers(min_value=-10, max_value=10),
)
small_nonzero_finite_decimals = small_finite_decimals.filter(lambda d: d != 0)


class TestNeg:
    @given(d=finite_decimals)
    @settings(max_examples=200)
    def test_neg_matches_cpython(self, d):
        s, i, e, sp = cpython_to_model(d)
        m = ModelDecimal(s, i, e, sp)
        result = m.__neg__()
        expected = cpython_to_model(-d)
        assert values_equal(result, expected)

    @given(d=finite_decimals)
    @settings(max_examples=200)
    def test_double_neg_roundtrip(self, d):
        s, i, e, sp = cpython_to_model(d)
        m = ModelDecimal(s, i, e, sp)
        result = m.__neg__().__neg__()
        assert result._sign == m._sign
        assert result._int == m._int
        assert result._exp == m._exp
        assert result._is_special == m._is_special

    def test_nan_neg_preserves_nan(self):
        nan = ModelDecimal(0, 0, 0, 2)
        result = nan.__neg__()
        assert result._is_special == 2


class TestAbs:
    @given(d=finite_decimals)
    @settings(max_examples=200)
    def test_abs_matches_cpython(self, d):
        s, i, e, sp = cpython_to_model(d)
        m = ModelDecimal(s, i, e, sp)
        result = m.__abs__()
        expected = cpython_to_model(abs(d))
        assert (result._sign, result._int, result._exp, result._is_special) == expected

    @given(d=finite_decimals)
    @settings(max_examples=200)
    def test_abs_always_non_negative(self, d):
        s, i, e, sp = cpython_to_model(d)
        m = ModelDecimal(s, i, e, sp)
        assert m.__abs__()._sign == 0


class TestAdd:
    @given(a=finite_decimals, b=finite_decimals)
    @settings(max_examples=200)
    def test_add_matches_cpython(self, a, b):
        with cpython_decimal.localcontext() as ctx:
            ctx.traps[cpython_decimal.Inexact] = False
            cpython_result = a + b
            assume(not ctx.flags[cpython_decimal.Inexact])
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        result = ma.__add__(mb)
        expected = cpython_to_model(cpython_result)
        assert values_equal(result, expected)

    @given(d=finite_decimals)
    @settings(max_examples=200)
    def test_add_zero_identity(self, d):
        s, i, e, sp = cpython_to_model(d)
        m = ModelDecimal(s, i, e, sp)
        zero = ModelDecimal(0, 0, 0, 0)
        result = m.__add__(zero)
        assert m.__eq__(result) or (m._int == 0 and result._int == 0)

    def test_inf_plus_neg_inf_is_nan(self):
        pos_inf = ModelDecimal(0, 0, 0, 1)
        neg_inf = ModelDecimal(1, 0, 0, 1)
        result = pos_inf.__add__(neg_inf)
        assert result._is_special == 2

    def test_nan_propagation(self):
        nan = ModelDecimal(0, 0, 0, 2)
        val = ModelDecimal(0, 1, 0, 0)
        assert nan.__add__(val)._is_special == 2
        assert val.__add__(nan)._is_special == 2


class TestSub:
    @given(a=finite_decimals, b=finite_decimals)
    @settings(max_examples=200)
    def test_sub_matches_cpython(self, a, b):
        with cpython_decimal.localcontext() as ctx:
            ctx.traps[cpython_decimal.Inexact] = False
            cpython_result = a - b
            assume(not ctx.flags[cpython_decimal.Inexact])
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        result = ma.__sub__(mb)
        expected = cpython_to_model(cpython_result)
        assert values_equal(result, expected)


class TestMul:
    @given(a=finite_decimals, b=finite_decimals)
    @settings(max_examples=200)
    def test_mul_matches_cpython(self, a, b):
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        result = ma.__mul__(mb)
        expected = cpython_to_model(a * b)
        assert (result._sign, result._int, result._exp, result._is_special) == expected

    def test_inf_times_zero_is_nan(self):
        inf = ModelDecimal(0, 0, 0, 1)
        zero = ModelDecimal(0, 0, 0, 0)
        assert inf.__mul__(zero)._is_special == 2
        assert zero.__mul__(inf)._is_special == 2


class TestTrueDiv:
    @given(a=finite_decimals, b=nonzero_finite_decimals)
    @settings(max_examples=200)
    def test_truediv_sign_matches_cpython(self, a, b):
        """Sign of result matches CPython (exact coefficient may differ due to precision)."""
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        result = ma.__truediv__(mb)
        cpython_result = a / b
        cp_sign, _, _, _ = cpython_to_model(cpython_result)
        if result._int == 0:
            pass  # zero can have either sign
        else:
            assert result._sign == cp_sign

    def test_zero_div_zero_is_nan(self):
        zero = ModelDecimal(0, 0, 0, 0)
        assert zero.__truediv__(zero)._is_special == 2

    def test_x_div_zero_is_inf(self):
        x = ModelDecimal(0, 5, 0, 0)
        zero = ModelDecimal(0, 0, 0, 0)
        result = x.__truediv__(zero)
        assert result._is_special == 1

    def test_inf_div_inf_is_nan(self):
        inf = ModelDecimal(0, 0, 0, 1)
        assert inf.__truediv__(inf)._is_special == 2


class TestFloorDiv:
    @given(a=small_finite_decimals, b=small_nonzero_finite_decimals)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.filter_too_much])
    def test_floordiv_matches_cpython(self, a, b):
        with cpython_decimal.localcontext() as ctx:
            ctx.traps[cpython_decimal.InvalidOperation] = False
            cpython_result = a // b
            assume(not cpython_result.is_nan())
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        result = ma.__floordiv__(mb)
        expected = cpython_to_model(cpython_result)
        assert values_equal(result, expected)


class TestMod:
    @given(a=small_finite_decimals, b=small_nonzero_finite_decimals)
    @settings(max_examples=200, suppress_health_check=[HealthCheck.filter_too_much])
    def test_mod_matches_cpython(self, a, b):
        with cpython_decimal.localcontext() as ctx:
            ctx.traps[cpython_decimal.InvalidOperation] = False
            cpython_result = a % b
            assume(not cpython_result.is_nan())
        sa, ia, ea, spa = cpython_to_model(a)
        sb, ib, eb, spb = cpython_to_model(b)
        ma = ModelDecimal(sa, ia, ea, spa)
        mb = ModelDecimal(sb, ib, eb, spb)
        result = ma.__mod__(mb)
        expected = cpython_to_model(cpython_result)
        assert values_equal(result, expected)
