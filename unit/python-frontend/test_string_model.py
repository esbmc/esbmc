"""Differential tests validating the string model constants against CPython's string module.

models/string.py provides the character-class constants of the stdlib ``string``
module. Each must equal its CPython counterpart exactly.
"""

import importlib.util
import os
import string as cpython_string

import pytest


def load_string_model():
    """Load models/string.py as a standalone module."""
    model_path = os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "..", "..", "src", "python-frontend", "models", "string.py"
        )
    )
    spec = importlib.util.spec_from_file_location("string_model", model_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_model = load_string_model()


CONSTANTS = [
    "digits",
    "octdigits",
    "ascii_lowercase",
    "ascii_uppercase",
    "ascii_letters",
    "hexdigits",
    "punctuation",
    "whitespace",
    "printable",
]


class TestStringConstants:
    @pytest.mark.parametrize("name", CONSTANTS)
    def test_constant_matches_cpython(self, name):
        assert getattr(_model, name) == getattr(cpython_string, name)

    def test_printable_is_concatenation(self):
        # CPython defines printable = digits + ascii_letters + punctuation + whitespace.
        expected = (_model.digits + _model.ascii_letters
                    + _model.punctuation + _model.whitespace)
        assert _model.printable == expected
