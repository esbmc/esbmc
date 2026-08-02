import ast
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY_FRONTEND_DIR = os.path.join(ROOT, "src", "python-frontend")

if PY_FRONTEND_DIR not in sys.path:
    sys.path.insert(0, PY_FRONTEND_DIR)


# The local frontend `parser` package (added to sys.path above) shadows the
# removed stdlib `parser` module; deprecated-module is a false positive here.
# pylint: disable=wrong-import-position,deprecated-module
from parser import parser as parser_mod
import preprocessor as preprocessor_mod


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def test_parser_add_type_annotation_for_complex_constant():
    assign = ast.parse("x = 2j").body[0]

    parser_mod.add_type_annotation(assign)

    require(isinstance(assign.value, ast.Constant), "expected ast.Constant")
    require(
        getattr(assign.value, "esbmc_type_annotation", None) == "complex",
        "expected complex annotation",
    )
    require(getattr(assign.value, "real_value", None) == 0.0, "expected real=0.0")
    require(getattr(assign.value, "imag_value", None) == 2.0, "expected imag=2.0")


def test_preprocessor_infers_complex_constant_type():
    pre = preprocessor_mod.Preprocessor("test_module")
    node = ast.Constant(value=complex(3.0, -4.5))

    inferred = pre._infer_type_from_constant(node)

    require(inferred == "complex", "expected inferred type complex")
