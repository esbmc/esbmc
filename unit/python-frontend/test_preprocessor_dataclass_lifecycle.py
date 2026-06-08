"""Tests for dataclass lifecycle support: InitVar, ClassVar and __post_init__."""

import ast
import importlib.util
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY_FRONTEND_DIR = os.path.join(ROOT, "src", "python-frontend")

if PY_FRONTEND_DIR not in sys.path:
    sys.path.insert(0, PY_FRONTEND_DIR)


def _load_module(module_name, rel_path):
    spec = importlib.util.spec_from_file_location(
        module_name, os.path.join(ROOT, rel_path)
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


preprocessor_mod = _load_module(
    "esbmc_preprocessor_dataclass_lifecycle", "src/python-frontend/preprocessor.py"
)


def _make_pre():
    return preprocessor_mod.Preprocessor("test_module")


def _transform(src):
    return _make_pre().visit(ast.parse(src))


def _get_class(module, name):
    return next(
        (s for s in module.body if isinstance(s, ast.ClassDef) and s.name == name),
        None,
    )


def _get_init(cls):
    return next(
        (s for s in cls.body if isinstance(s, ast.FunctionDef) and s.name == "__init__"),
        None,
    )


def _get_metadata_assign(cls):
    return next(
        (
            s
            for s in cls.body
            if isinstance(s, ast.Assign)
            and len(s.targets) == 1
            and isinstance(s.targets[0], ast.Name)
            and s.targets[0].id == "__dataclass_fields__"
        ),
        None,
    )


def test_initvar_becomes_init_param_and_post_init_argument():
    src = (
        "from dataclasses import dataclass, InitVar\n"
        "@dataclass\n"
        "class Config:\n"
        "    name: str\n"
        "    seed: InitVar[int]\n"
        "    priority: int = 0\n"
        "    def __post_init__(self, seed):\n"
        "        self.priority = seed\n"
    )
    module = _transform(src)
    cls = _get_class(module, "Config")
    init = _get_init(cls)
    assert init is not None

    arg_names = [arg.arg for arg in init.args.args]
    assert arg_names == ["self", "name", "seed", "priority"]
    assert len(init.args.defaults) == 1
    assert isinstance(init.args.defaults[0], ast.Constant)
    assert init.args.defaults[0].value == 0

    assigned_attrs = [
        stmt.targets[0].attr
        for stmt in init.body
        if isinstance(stmt, ast.Assign)
        and isinstance(stmt.targets[0], ast.Attribute)
    ]
    assert assigned_attrs == ["name", "priority"]

    post_init_stmt = init.body[-1]
    assert isinstance(post_init_stmt, ast.Expr)
    post_init_call = post_init_stmt.value
    assert isinstance(post_init_call, ast.Call)
    assert isinstance(post_init_call.func, ast.Attribute)
    assert post_init_call.func.attr == "__post_init__"
    assert isinstance(post_init_call.func.value, ast.Name)
    assert post_init_call.func.value.id == "Config"
    assert [arg.id for arg in post_init_call.args] == ["self", "seed"]


def test_classvar_is_excluded_from_init_and_metadata():
    src = (
        "from dataclasses import dataclass\n"
        "from typing import ClassVar\n"
        "@dataclass\n"
        "class Config:\n"
        "    version: ClassVar[int] = 3\n"
        "    name: str\n"
    )
    module = _transform(src)
    cls = _get_class(module, "Config")
    init = _get_init(cls)
    assert init is not None

    arg_names = [arg.arg for arg in init.args.args]
    assert arg_names == ["self", "name"]

    metadata = _get_metadata_assign(cls)
    assert metadata is not None
    assert isinstance(metadata.value, ast.Tuple)
    field_names = [elt.value for elt in metadata.value.elts]
    assert field_names == ["name"]

    classvar_annassign = next(
        (
            stmt
            for stmt in cls.body
            if isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "version"
        ),
        None,
    )
    assert classvar_annassign is not None


def test_post_init_only_dataclass_still_gets_synthesized_init():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Hook:\n"
        "    def __post_init__(self):\n"
        "        self.ready = True\n"
    )
    module = _transform(src)
    cls = _get_class(module, "Hook")
    init = _get_init(cls)
    assert init is not None
    assert [arg.arg for arg in init.args.args] == ["self"]
    assert len(init.body) == 1
    assert isinstance(init.body[0], ast.Expr)
    assert isinstance(init.body[0].value.func.value, ast.Name)
    assert init.body[0].value.func.value.id == "Hook"


def test_inherited_post_init_still_triggers_guarded_call():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Base:\n"
        "    def __post_init__(self):\n"
        "        self.base_ready = True\n"
        "@dataclass\n"
        "class Child(Base):\n"
        "    value: int\n"
    )
    module = _transform(src)
    child = _get_class(module, "Child")
    init = _get_init(child)
    assert init is not None
    assert isinstance(init.body[-1], ast.Expr)
    post_init_call = init.body[-1].value
    assert isinstance(post_init_call, ast.Call)
    assert isinstance(post_init_call.func, ast.Attribute)
    assert post_init_call.func.attr == "__post_init__"
    assert isinstance(post_init_call.func.value, ast.Name)
    assert post_init_call.func.value.id == "self"
    assert len(post_init_call.args) == 0


def test_incompatible_post_init_signature_raises_syntax_error():
    src = (
        "from dataclasses import dataclass, InitVar\n"
        "@dataclass\n"
        "class Bad:\n"
        "    seed: InitVar[int]\n"
        "    def __post_init__(self):\n"
        "        pass\n"
    )
    with pytest.raises(SyntaxError, match="incompatible __post_init__ signature"):
        _transform(src)


def test_post_init_with_extra_required_positional_parameter_is_rejected():
    src = (
        "from dataclasses import dataclass, InitVar\n"
        "@dataclass\n"
        "class Bad:\n"
        "    seed: InitVar[int]\n"
        "    def __post_init__(self, seed, extra):\n"
        "        pass\n"
    )
    with pytest.raises(SyntaxError, match="incompatible __post_init__ signature"):
        _transform(src)


def test_required_keyword_only_post_init_parameter_is_rejected():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Bad:\n"
        "    value: int\n"
        "    def __post_init__(self, *, extra):\n"
        "        pass\n"
    )
    with pytest.raises(
        SyntaxError, match="required keyword-only parameters are not supported"
    ):
        _transform(src)