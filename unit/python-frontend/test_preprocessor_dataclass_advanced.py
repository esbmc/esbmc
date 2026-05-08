"""Advanced dataclass tests for Marco F (flags and inheritance)."""

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
    "esbmc_preprocessor_dataclass_advanced", "src/python-frontend/preprocessor.py"
)


def _transform(src):
    return preprocessor_mod.Preprocessor("test_module").visit(ast.parse(src))


def _get_class(module, name):
    return next(
        (s for s in module.body if isinstance(s, ast.ClassDef) and s.name == name),
        None,
    )


def _get_method(cls, name):
    return next(
        (s for s in cls.body if isinstance(s, ast.FunctionDef) and s.name == name),
        None,
    )


def test_class_flags_generate_derived_methods():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass(order=True, frozen=True)\n"
        "class P:\n"
        "    x: int\n"
        "    y: int\n"
    )
    module = _transform(src)
    cls = _get_class(module, "P")

    assert _get_method(cls, "__init__") is not None
    assert _get_method(cls, "__repr__") is not None
    assert _get_method(cls, "__eq__") is not None
    assert _get_method(cls, "__hash__") is not None
    assert _get_method(cls, "__lt__") is not None
    assert _get_method(cls, "__le__") is not None
    assert _get_method(cls, "__gt__") is not None
    assert _get_method(cls, "__ge__") is not None


def test_kw_only_class_flag_moves_fields_to_kwonlyargs():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass(kw_only=True)\n"
        "class C:\n"
        "    x: int\n"
        "    y: int = 1\n"
    )
    module = _transform(src)
    cls = _get_class(module, "C")
    init = _get_method(cls, "__init__")

    assert [a.arg for a in init.args.args] == ["self"]
    assert [a.arg for a in init.args.kwonlyargs] == ["x", "y"]
    assert init.args.kw_defaults[0] is None
    assert isinstance(init.args.kw_defaults[1], ast.Constant)
    assert init.args.kw_defaults[1].value == 1


def test_field_flags_affect_init_and_metadata_semantics():
    src = (
        "from dataclasses import dataclass, field\n"
        "@dataclass\n"
        "class C:\n"
        "    x: int\n"
        "    y: int = field(init=False, default=7, repr=False, compare=False)\n"
    )
    module = _transform(src)
    cls = _get_class(module, "C")
    init = _get_method(cls, "__init__")

    assert [a.arg for a in init.args.args] == ["self", "x"]
    assigns = [
        s for s in init.body
        if isinstance(s, ast.Assign)
        and len(s.targets) == 1
        and isinstance(s.targets[0], ast.Attribute)
    ]
    names = [s.targets[0].attr for s in assigns]
    assert names == ["x", "y"]


def test_inheritance_overrides_preserve_position():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Base:\n"
        "    a: int\n"
        "    b: int = 1\n"
        "@dataclass\n"
        "class Child(Base):\n"
        "    b: int = 9\n"
        "    c: int = 3\n"
    )
    module = _transform(src)
    cls = _get_class(module, "Child")
    init = _get_method(cls, "__init__")

    assert [a.arg for a in init.args.args] == ["self", "a", "b", "c"]
    assert [d.value for d in init.args.defaults] == [9, 3]


def test_invalid_decorator_option_is_rejected():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass(unknown=True)\n"
        "class C:\n"
        "    x: int\n"
    )
    with pytest.raises(SyntaxError, match="unsupported dataclass option"):
        _transform(src)


def test_order_requires_eq():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass(order=True, eq=False)\n"
        "class C:\n"
        "    x: int\n"
    )
    with pytest.raises(SyntaxError, match="order=True"):
        _transform(src)


def test_slots_true_rejects_explicit_slots_definition():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass(slots=True)\n"
        "class C:\n"
        "    __slots__ = ('x',)\n"
        "    x: int\n"
    )
    with pytest.raises(SyntaxError, match="slots=True"):
        _transform(src)


def test_unsafe_hash_true_rejects_explicit_hash_method():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass(unsafe_hash=True)\n"
        "class C:\n"
        "    x: int\n"
        "    def __hash__(self):\n"
        "        return 1\n"
    )
    with pytest.raises(SyntaxError, match="unsafe_hash=True"):
        _transform(src)


def test_unsafe_hash_true_rejects_explicit_hash_attribute():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass(unsafe_hash=True)\n"
        "class C:\n"
        "    x: int\n"
        "    __hash__ = None\n"
    )
    with pytest.raises(SyntaxError, match="unsafe_hash=True"):
        _transform(src)


def test_invalid_field_option_is_rejected():
    src = (
        "from dataclasses import dataclass, field\n"
        "@dataclass\n"
        "class C:\n"
        "    x: int = field(foo=1)\n"
    )
    with pytest.raises(SyntaxError, match="unsupported dataclass field option"):
        _transform(src)
