"""Differential tests validating the int model against CPython's int.

models/int.py models the no-argument int methods as ``@classmethod``s taking the
receiver value as the first value parameter (CPython auto-binds ``cls``, so the
differential call passes the value only). bit_length/bit_count are annotated with
the ESBMC-internal ``IntWide`` type, which is not a Python symbol, so it is injected
into the module namespace before the class body executes (see int.py:5-8).

Scope notes (model semantics that bound what is differentially comparable):
- bit_length loops while ``n > 0``, so it matches CPython only for n >= 0 (CPython
  reports the bit length of the magnitude; the model returns 0 for negatives).
- bit_count folds the magnitude first, so it matches CPython for every sign.
- from_bytes on empty bytes with signed=True indexes out of range in the model
  (CPython returns 0); that input is left out of the differential strategy and the
  empty case is covered unsigned, where the model returns 0 as CPython does.
"""

import importlib.util
import os

from hypothesis import given, settings, strategies as st


def load_int_model():
    """Load the int class from models/int.py, injecting the IntWide annotation."""
    model_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "src", "python-frontend", "models", "int.py"
        )
    )
    spec = importlib.util.spec_from_file_location("int_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    # `IntWide` annotates bit_length/bit_count parameters and is evaluated when the
    # class body runs (no `from __future__ import annotations`). It is not a Python
    # symbol; ESBMC maps it to a wide bitvector. Bind it to int so the import works.
    mod.IntWide = int
    spec.loader.exec_module(mod)
    return getattr(mod, "int")


ModelInt = load_int_model()


# bit_length is bounded by a 512-shift cap; keep magnitudes well under 2**512.
nonneg = st.integers(min_value=0, max_value=2**256)
signed_ints = st.integers(min_value=-(2**256), max_value=2**256)
byte_blobs = st.binary(min_size=1, max_size=8)


class TestBitLength:
    @given(n=nonneg)
    @settings(max_examples=400)
    def test_matches_cpython(self, n):
        assert ModelInt.bit_length(n) == n.bit_length()

    def test_zero(self):
        assert ModelInt.bit_length(0) == (0).bit_length()


class TestBitCount:
    @given(n=signed_ints)
    @settings(max_examples=400)
    def test_matches_cpython_any_sign(self, n):
        assert ModelInt.bit_count(n) == n.bit_count()

    def test_zero(self):
        assert ModelInt.bit_count(0) == (0).bit_count()


class TestConjugate:
    @given(n=signed_ints)
    @settings(max_examples=200)
    def test_is_identity(self, n):
        assert ModelInt.conjugate(n) == n.conjugate()


class TestFromBytes:
    @given(data=byte_blobs, big_endian=st.booleans(), signed=st.booleans())
    @settings(max_examples=400)
    def test_matches_cpython(self, data, big_endian, signed):
        byteorder = "big" if big_endian else "little"
        expected = int.from_bytes(data, byteorder, signed=signed)
        assert ModelInt.from_bytes(data, big_endian, signed) == expected

    @given(big_endian=st.booleans())
    def test_empty_unsigned_is_zero(self, big_endian):
        assert ModelInt.from_bytes(b"", big_endian, False) == 0
