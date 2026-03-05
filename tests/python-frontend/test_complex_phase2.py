import ast
import importlib.util
import os
import sys


ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY_FRONTEND_DIR = os.path.join(ROOT, "src", "python-frontend")

if PY_FRONTEND_DIR not in sys.path:
    sys.path.insert(0, PY_FRONTEND_DIR)


def _load_module(module_name: str, rel_path: str):
    module_path = os.path.join(ROOT, rel_path)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


parser_mod = _load_module("esbmc_parser", "src/python-frontend/parser.py")
preprocessor_mod = _load_module("esbmc_preprocessor", "src/python-frontend/preprocessor.py")


def test_parser_add_type_annotation_for_complex_constant():
    assign = ast.parse("x = 2j").body[0]

    parser_mod.add_type_annotation(assign)

    assert isinstance(assign.value, ast.Constant)
    assert getattr(assign.value, "esbmc_type_annotation", None) == "complex"
    assert getattr(assign.value, "real_value", None) == 0.0
    assert getattr(assign.value, "imag_value", None) == 2.0


def test_preprocessor_infers_complex_constant_type():
    pre = preprocessor_mod.Preprocessor("test_module")
    node = ast.Constant(value=complex(3.0, -4.5))

    inferred = pre._infer_type_from_constant(node)

    assert inferred == "complex"
