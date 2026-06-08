"""Tests for dataclass support for ``field()`` defaults and factories."""

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
    "esbmc_preprocessor_dataclass_defaults", "src/python-frontend/preprocessor.py"
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


# ---------------------------------------------------------------------------
# 1. Raw scalar defaults (``x: int = 5``)
# ---------------------------------------------------------------------------

def test_raw_scalar_default_becomes_init_default():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class C:\n"
        "    x: int\n"
        "    y: int = 5\n"
    )
    module = _transform(src)
    cls = _get_class(module, "C")
    assert cls is not None

    init = _get_init(cls)
    assert init is not None, "__init__ must be synthesized"

    arg_names = [a.arg for a in init.args.args]
    assert arg_names == ["self", "x", "y"]

    # One default for the trailing ``y`` argument.
    assert len(init.args.defaults) == 1
    default = init.args.defaults[0]
    assert isinstance(default, ast.Constant) and default.value == 5


# ---------------------------------------------------------------------------
# 2. ``field(default=...)`` literal default
# ---------------------------------------------------------------------------

def test_field_default_keyword_extracted_as_init_default():
    src = (
        "from dataclasses import dataclass, field\n"
        "@dataclass\n"
        "class C:\n"
        "    x: int = field(default=42)\n"
    )
    module = _transform(src)
    cls = _get_class(module, "C")
    init = _get_init(cls)
    assert init is not None

    assert len(init.args.defaults) == 1
    default = init.args.defaults[0]
    assert isinstance(default, ast.Constant) and default.value == 42


def test_field_call_with_no_default_or_factory_is_required():
    """``x: int = field()`` declares a required field.

    ``_parse_field_call`` returns ``(None, None)`` for an empty ``field()``
    call, which means the synthesized __init__ must treat the attribute as a
    required positional parameter (no entry in ``defaults``).
    """
    src = (
        "from dataclasses import dataclass, field\n"
        "@dataclass\n"
        "class C:\n"
        "    x: int = field()\n"
    )
    module = _transform(src)
    cls = _get_class(module, "C")
    init = _get_init(cls)
    assert init is not None

    arg_names = [a.arg for a in init.args.args]
    assert arg_names == ["self", "x"]
    assert init.args.defaults == []


# ---------------------------------------------------------------------------
# 3. ``field(default_factory=...)`` desugars to direct in-body factory call
# ---------------------------------------------------------------------------

def test_field_default_factory_assigns_directly_in_body():
    src = (
        "from dataclasses import dataclass, field\n"
        "from typing import List\n"
        "@dataclass\n"
        "class C:\n"
        "    items: List[int] = field(default_factory=list)\n"
    )
    module = _transform(src)

    cls = _get_class(module, "C")
    init = _get_init(cls)
    assert init is not None

    # Marco F: factory-backed fields are now exposed as overridable init
    # parameters (defaulting to None => call factory).
    arg_names = [a.arg for a in init.args.args]
    assert arg_names == ["self", "items"]
    assert len(init.args.defaults) == 1
    assert isinstance(init.args.defaults[0], ast.Constant)
    assert init.args.defaults[0].value is None

    # Body must contain a single Assign using the factory directly.
    assert len(init.body) == 1
    assign = init.body[0]
    assert isinstance(assign, ast.Assign)
    assert len(assign.targets) == 1
    target = assign.targets[0]
    assert isinstance(target, ast.Attribute) and target.attr == "items"
    assert isinstance(assign.value, ast.Call)
    assert isinstance(assign.value.func, ast.Name)
    assert assign.value.func.id == "list"
    assert assign.value.args == [] and assign.value.keywords == []


def test_no_module_level_factory_sentinel_injected():
    src = (
        "from dataclasses import dataclass, field\n"
        "@dataclass\n"
        "class C:\n"
        "    items: list = field(default_factory=list)\n"
    )
    module = _transform(src)
    # The desugaring must remain purely local to __init__: no module-level
    # helper symbol should ever be injected.
    assert not any(
        isinstance(s, ast.Assign)
        and isinstance(s.targets[0], ast.Name)
        and s.targets[0].id.startswith("__esbmc_dc_")
        for s in module.body
    )


# ---------------------------------------------------------------------------
# 4. Required + defaulted fields preserve positional order
# ---------------------------------------------------------------------------

def test_required_then_defaulted_positional_ordering():
    src = (
        "from dataclasses import dataclass, field\n"
        "@dataclass\n"
        "class Task:\n"
        "    name: str\n"
        "    priority: int = 0\n"
        "    tags: list = field(default_factory=list)\n"
    )
    module = _transform(src)
    cls = _get_class(module, "Task")
    init = _get_init(cls)
    assert init is not None

    # Factory field is now overridable in __init__ (default None -> factory).
    arg_names = [a.arg for a in init.args.args]
    assert arg_names == ["self", "name", "priority", "tags"]
    # Two trailing defaults: priority=0 and tags=None.
    assert len(init.args.defaults) == 2
    assert (
        isinstance(init.args.defaults[0], ast.Constant)
        and init.args.defaults[0].value == 0
    )
    assert isinstance(init.args.defaults[1], ast.Constant)
    assert init.args.defaults[1].value is None


# ---------------------------------------------------------------------------
# 5. Field declarations are stripped from the class body
# ---------------------------------------------------------------------------

def test_defaulted_field_annassign_stripped_from_class_body():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class C:\n"
        "    x: int = 1\n"
        "    y: int = 2\n"
    )
    module = _transform(src)
    cls = _get_class(module, "C")

    # No leftover AnnAssign for x/y at class scope.
    leftover = [
        s for s in cls.body
        if isinstance(s, ast.AnnAssign)
        and isinstance(s.target, ast.Name)
        and s.target.id in {"x", "y"}
    ]
    assert leftover == [], (
        f"AnnAssigns for defaulted fields should be stripped, got {leftover}"
    )


# ---------------------------------------------------------------------------
# 6. Validation: required field after defaulted field must raise
# ---------------------------------------------------------------------------

def test_non_default_field_after_default_raises_syntax_error():
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Bad:\n"
        "    x: int = 1\n"
        "    y: int\n"
    )
    with pytest.raises(SyntaxError):
        _transform(src)


# ---------------------------------------------------------------------------
# 7. Self-assignments preserved for every field, including factory ones
# ---------------------------------------------------------------------------

def test_init_body_assigns_every_field_to_self():
    src = (
        "from dataclasses import dataclass, field\n"
        "@dataclass\n"
        "class C:\n"
        "    a: int\n"
        "    b: int = 7\n"
        "    c: list = field(default_factory=list)\n"
    )
    module = _transform(src)
    cls = _get_class(module, "C")
    init = _get_init(cls)

    assigned_attrs = []
    for stmt in init.body:
        if (
            isinstance(stmt, ast.Assign)
            and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Attribute)
            and isinstance(stmt.targets[0].value, ast.Name)
            and stmt.targets[0].value.id == "self"
        ):
            assigned_attrs.append(stmt.targets[0].attr)
    assert assigned_attrs == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# 8. Aliased imports of ``dataclass`` and ``field`` are recognized
# ---------------------------------------------------------------------------

def test_dataclass_decorator_alias_is_recognized():
    src = (
        "from dataclasses import dataclass as dc\n"
        "@dc\n"
        "class C:\n"
        "    x: int = 5\n"
    )
    module = _transform(src)
    cls = _get_class(module, "C")
    init = _get_init(cls)
    assert init is not None, "aliased @dc should still trigger __init__ synthesis"
    arg_names = [a.arg for a in init.args.args]
    assert arg_names == ["self", "x"]
    assert len(init.args.defaults) == 1


def test_field_alias_default_is_extracted():
    src = (
        "from dataclasses import dataclass, field as f\n"
        "@dataclass\n"
        "class C:\n"
        "    x: int = f(default=42)\n"
    )
    module = _transform(src)
    cls = _get_class(module, "C")
    init = _get_init(cls)
    assert init is not None
    arg_names = [a.arg for a in init.args.args]
    assert arg_names == ["self", "x"]
    assert len(init.args.defaults) == 1
    default_node = init.args.defaults[0]
    assert isinstance(default_node, ast.Constant) and default_node.value == 42


def test_field_alias_default_factory_emits_body_assignment():
    src = (
        "from dataclasses import dataclass, field as f\n"
        "@dataclass\n"
        "class C:\n"
        "    items: list = f(default_factory=list)\n"
    )
    module = _transform(src)
    cls = _get_class(module, "C")
    init = _get_init(cls)
    assert init is not None
    # Factory field is exposed as overridable parameter with None default.
    assert [a.arg for a in init.args.args] == ["self", "items"]
    assert len(init.args.defaults) == 1
    assert isinstance(init.args.defaults[0], ast.Constant)
    assert init.args.defaults[0].value is None
    # And must be assigned via direct factory call.
    factory_assigns = [
        s
        for s in init.body
        if isinstance(s, ast.Assign)
        and len(s.targets) == 1
        and isinstance(s.targets[0], ast.Attribute)
        and s.targets[0].attr == "items"
        and isinstance(s.value, ast.Call)
        and isinstance(s.value.func, ast.Name)
        and s.value.func.id == "list"
    ]
    assert len(factory_assigns) == 1


# ---------------------------------------------------------------------------
# 9. SyntaxError for invalid field ordering carries the class line number
# ---------------------------------------------------------------------------

def test_non_default_after_default_error_includes_lineno():
    src = (
        "# pad\n"
        "# pad\n"
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Bad:\n"
        "    x: int = 1\n"
        "    y: int\n"
    )
    with pytest.raises(SyntaxError) as exc_info:
        _transform(src)
    msg = str(exc_info.value)
    assert "Bad" in msg
    assert "'y'" in msg
    # ``class Bad:`` is on line 5 of the source above.
    assert "line 5" in msg


def test_unsupported_default_expression_raises_syntax_error():
    src = (
        "from dataclasses import dataclass\n"
        "def mk():\n"
        "    return 5\n"
        "@dataclass\n"
        "class C:\n"
        "    x: int = mk()\n"
    )
    with pytest.raises(SyntaxError) as exc_info:
        _transform(src)
    assert "unsupported dataclass default expression" in str(exc_info.value)
