"""Differential tests validating the float model against CPython's float.

models/float.py models ``float`` as a class whose methods are ``@classmethod``s
taking the encapsulated value as the first value parameter; the frontend lowers
``x.is_integer()`` into ``is_integer(x)``. Under CPython the classmethod auto-binds
``cls``, so the differential call passes the value only: ``Model.is_integer(x)``.
"""

import importlib.util
import os

from hypothesis import given, settings, strategies as st


def load_float_model():
    """Load the float class from models/float.py."""
    model_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "src", "python-frontend", "models", "float.py"
        )
    )
    spec = importlib.util.spec_from_file_location("float_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.float


ModelFloat = load_float_model()


# Finite floats only: the model evaluates ``x == int(x)`` and int(nan)/int(inf)
# raise, so non-finite inputs are out of the model's domain (and CPython's
# float.is_integer simply returns False for them).
finite_floats = st.floats(allow_nan=False, allow_infinity=False)
# A generous helping of exact integral floats, where the True branch matters.
integral_floats = st.integers(min_value=-10**12, max_value=10**12).map(float)


class TestIsInteger:
    @given(x=finite_floats)
    @settings(max_examples=400)
    def test_matches_cpython(self, x):
        assert ModelFloat.is_integer(x) == x.is_integer()

    @given(x=integral_floats)
    @settings(max_examples=200)
    def test_integral_values_are_integer(self, x):
        assert ModelFloat.is_integer(x) is True
        assert x.is_integer() is True

    @given(n=st.integers(min_value=-10**9, max_value=10**9),
           frac=st.floats(min_value=0.01, max_value=0.99))
    @settings(max_examples=200)
    def test_non_integral_values_are_not_integer(self, n, frac):
        x = n + frac
        assert ModelFloat.is_integer(x) == x.is_integer()
        assert ModelFloat.is_integer(x) is False
