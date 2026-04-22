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


preprocessor_mod = _load_module("esbmc_preprocessor_regressions",
                                "src/python-frontend/preprocessor.py")


def test_isolate_genexp_targets_renames_inner_iter_loads_before_shadowing():
    call = ast.parse("all(x > 0 for x in xs for x in range(x))", mode="eval").body
    genexp = call.args[0]

    pre = preprocessor_mod.Preprocessor("test_module")
    isolated = pre._isolate_genexp_targets(genexp)

    outer_target = isolated.generators[0].target
    inner_iter_arg = isolated.generators[1].iter.args[0]
    inner_target = isolated.generators[1].target

    assert isinstance(outer_target, ast.Name)
    assert isinstance(inner_iter_arg, ast.Name)
    assert isinstance(inner_target, ast.Name)
    assert inner_iter_arg.id == outer_target.id
    assert inner_target.id != outer_target.id


def test_lower_listcomp_in_boolop_preserves_list_result_type():
    module = ast.parse(
        """
xs: list[int] = [1]
vals = [x for x in xs] and []
if vals:
    pass
"""
    )

    pre = preprocessor_mod.Preprocessor("test_module")
    transformed = pre.visit(module)

    assert pre.known_variable_types["vals"] == "list"
    if_node = transformed.body[-1]
    assert isinstance(if_node, ast.If)
    assert isinstance(if_node.test, ast.Compare)
    assert isinstance(if_node.test.left, ast.Call)
    assert isinstance(if_node.test.left.func, ast.Name)
    assert if_node.test.left.func.id == "len"
