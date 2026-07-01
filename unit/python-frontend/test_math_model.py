"""Differential tests validating the pure-Python subset of the math model against CPython.

models/math.py mostly delegates to ESBMC intrinsics (``__ESBMC_*``) with no Python
binding — the transcendentals (sin, cos, exp, log, sqrt, ...) return None under
CPython and are out of scope here. This module tests only the functions whose body
is pure Python: the integer/combinatoric helpers plus floor/ceil/trunc, degrees/
radians and isclose.

floor/ceil guard on isinf/isnan, which delegate to ``__ESBMC_isinf``/``__ESBMC_isnan``;
those two intrinsics are bound to their CPython equivalents (math.isinf/math.isnan) so
the guards are runnable. No transcendental intrinsic is bound, so this stays a test of
the pure-Python logic only.
"""

import importlib.util
import math
import os

from hypothesis import assume, given, settings, strategies as st


def load_math_model():
    """Load models/math.py, binding the isinf/isnan intrinsics to CPython math."""
    model_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "src", "python-frontend", "models", "math.py"
        )
    )
    spec = importlib.util.spec_from_file_location("math_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    mod.__ESBMC_isinf = math.isinf
    mod.__ESBMC_isnan = math.isnan
    spec.loader.exec_module(mod)
    return mod


_m = load_math_model()


small_ints = st.integers(min_value=-10000, max_value=10000)
nonneg_small = st.integers(min_value=0, max_value=200)
finite_floats = st.floats(allow_nan=False, allow_infinity=False,
                          min_value=-1e15, max_value=1e15)


class TestIntegerHelpers:
    @given(a=small_ints, b=small_ints)
    @settings(max_examples=300)
    def test_gcd(self, a, b):
        assert _m.gcd(a, b) == math.gcd(a, b)

    @given(a=small_ints, b=small_ints)
    @settings(max_examples=300)
    def test_lcm(self, a, b):
        assert _m.lcm(a, b) == math.lcm(a, b)

    @given(n=nonneg_small)
    @settings(max_examples=200)
    def test_factorial(self, n):
        assert _m.factorial(n) == math.factorial(n)

    @given(n=nonneg_small, k=nonneg_small)
    @settings(max_examples=300)
    def test_comb(self, n, k):
        assert _m.comb(n, k) == math.comb(n, k)

    @given(n=nonneg_small, k=nonneg_small)
    @settings(max_examples=300)
    def test_perm_with_k(self, n, k):
        assert _m.perm(n, k) == math.perm(n, k)

    @given(n=nonneg_small)
    @settings(max_examples=200)
    def test_perm_default_k(self, n):
        # model perm(n) with default k == -1 returns factorial(n); math.perm(n) too.
        assert _m.perm(n) == math.perm(n)

    @given(n=st.integers(min_value=0, max_value=10**12))
    @settings(max_examples=300)
    def test_isqrt(self, n):
        assert _m.isqrt(n) == math.isqrt(n)

    @given(values=st.lists(st.integers(min_value=-50, max_value=50), max_size=12),
           start=st.integers(min_value=-10, max_value=10))
    @settings(max_examples=300)
    def test_prod(self, values, start):
        assert _m.prod(values, start) == math.prod(values, start=start)


class TestRoundingHelpers:
    @given(x=finite_floats)
    @settings(max_examples=400)
    def test_floor(self, x):
        assert _m.floor(x) == math.floor(x)

    @given(x=finite_floats)
    @settings(max_examples=400)
    def test_ceil(self, x):
        assert _m.ceil(x) == math.ceil(x)

    @given(x=finite_floats)
    @settings(max_examples=400)
    def test_trunc(self, x):
        assert _m.trunc(x) == math.trunc(x)


class TestFloatHelpers:
    @given(x=finite_floats)
    @settings(max_examples=300)
    def test_degrees(self, x):
        # Same formula and same pi as CPython -> bit-identical.
        assert _m.degrees(x) == math.degrees(x)

    @given(x=finite_floats)
    @settings(max_examples=300)
    def test_radians(self, x):
        assert _m.radians(x) == math.radians(x)

    @given(a=finite_floats, b=finite_floats)
    @settings(max_examples=400)
    def test_isclose_default_tolerances(self, a, b):
        assert _m.isclose(a, b) == math.isclose(a, b)

    @given(a=finite_floats, b=finite_floats,
           rel=st.floats(min_value=0.0, max_value=1.0),
           abs_=st.floats(min_value=0.0, max_value=1.0))
    @settings(max_examples=300)
    def test_isclose_custom_tolerances(self, a, b, rel, abs_):
        assume(not math.isnan(rel) and not math.isnan(abs_))
        assert _m.isclose(a, b, rel, abs_) == math.isclose(a, b, rel_tol=rel, abs_tol=abs_)

    def test_pi_matches_cpython(self):
        assert _m.pi == math.pi
