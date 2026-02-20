"""Property-based tests validating the Decimal model against CPython's decimal.Decimal."""

import sys
import os
import importlib.util
import decimal as cpython_decimal

import pytest
from hypothesis import given, strategies as st, assume, settings


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
    return mod.Decimal, mod._decimal_from_int


ModelDecimal, _decimal_from_int = load_model_decimal()


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
        assert m._is_special == 0


class TestDecimalFromInt:
    @given(n=st.integers(min_value=-10000, max_value=10000))
    @settings(max_examples=200)
    def test_from_int_matches_cpython(self, n):
        """_decimal_from_int produces same representation as CPython Decimal(n)."""
        m = _decimal_from_int(n)
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
