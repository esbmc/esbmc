"""Tests for type alias handling and dataclass docstring preservation in the preprocessor.

Covers three features added to preprocessor.py:
  1. _is_type_alias_expression(): detects Tuple[...]/List[...]/etc. as type alias RHS
  2. visit_Assign() type alias removal: strips alias assignments from runtime AST
  3. _resolve_annotation_aliases(): expands alias names in annotation contexts
  4. visit_AnnAssign(): resolves aliases in variable annotations
  5. visit_FunctionDef(): resolves aliases in return type and parameter annotations
  6. expand_dataclass(): inserts __init__ after docstring (not before it)
"""

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


preprocessor_mod = _load_module("esbmc_preprocessor_type_aliases",
                                "src/python-frontend/preprocessor.py")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pre():
    return preprocessor_mod.Preprocessor("test_module")


def _get_annotation_name(node):
    """Return the string id of an annotation Name node, or None."""
    if isinstance(node, ast.Name):
        return node.id
    return None


def _get_subscript_name(node):
    """Return the string id of the outer Name in a Subscript annotation."""
    if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name):
        return node.value.id
    return None


# ---------------------------------------------------------------------------
# 1. _is_type_alias_expression
# ---------------------------------------------------------------------------

def test_is_type_alias_expression_tuple():
    pre = _make_pre()
    value = ast.parse("Tuple[int, int]", mode="eval").body
    assert pre._is_type_alias_expression(value)


def test_is_type_alias_expression_list():
    pre = _make_pre()
    value = ast.parse("List[str]", mode="eval").body
    assert pre._is_type_alias_expression(value)


def test_is_type_alias_expression_optional():
    pre = _make_pre()
    value = ast.parse("Optional[int]", mode="eval").body
    assert pre._is_type_alias_expression(value)


def test_is_type_alias_expression_dict():
    pre = _make_pre()
    value = ast.parse("Dict[str, int]", mode="eval").body
    assert pre._is_type_alias_expression(value)


def test_is_type_alias_expression_union():
    pre = _make_pre()
    value = ast.parse("Union[int, str]", mode="eval").body
    assert pre._is_type_alias_expression(value)


def test_is_type_alias_expression_plain_name_is_false():
    pre = _make_pre()
    value = ast.parse("int", mode="eval").body
    assert not pre._is_type_alias_expression(value)


def test_is_type_alias_expression_call_is_false():
    pre = _make_pre()
    value = ast.parse("int()", mode="eval").body
    assert not pre._is_type_alias_expression(value)


def test_is_type_alias_expression_typing_attribute():
    """typing.Tuple[int, int] (attribute access) must also be recognized."""
    pre = _make_pre()
    value = ast.parse("typing.Tuple[int, int]", mode="eval").body
    assert pre._is_type_alias_expression(value)


# ---------------------------------------------------------------------------
# 2. visit_Assign – type alias assignments are removed from runtime AST
# ---------------------------------------------------------------------------

def test_type_alias_assignment_removed_from_ast():
    """Coordinate = Tuple[int, int] must not appear in the transformed module."""
    src = "from typing import Tuple\nCoordinate = Tuple[int, int]\n"
    module = ast.parse(src)
    pre = _make_pre()
    transformed = pre.visit(module)

    names = [
        stmt.targets[0].id
        for stmt in transformed.body
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
    ]
    assert "Coordinate" not in names, "type alias assignment should be removed"


def test_type_alias_stored_in_dict():
    """After visiting, the alias must be present in pre.type_aliases."""
    src = "from typing import Tuple\nCoordinate = Tuple[int, int]\n"
    module = ast.parse(src)
    pre = _make_pre()
    pre.visit(module)

    assert "Coordinate" in pre.type_aliases
    alias_node = pre.type_aliases["Coordinate"]
    assert _get_subscript_name(alias_node) == "Tuple"


def test_multiple_type_aliases_all_removed():
    src = (
        "from typing import Tuple, List\n"
        "Point = Tuple[int, int]\n"
        "Tags = List[str]\n"
        "x: int = 1\n"
    )
    module = ast.parse(src)
    pre = _make_pre()
    transformed = pre.visit(module)

    names = [
        stmt.targets[0].id
        for stmt in transformed.body
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
    ]
    assert "Point" not in names
    assert "Tags" not in names
    assert "Point" in pre.type_aliases
    assert "Tags" in pre.type_aliases


def test_normal_assignment_not_removed():
    """Regular x = 1 must NOT be removed."""
    src = "x = 1\n"
    module = ast.parse(src)
    pre = _make_pre()
    transformed = pre.visit(module)

    names = [
        stmt.targets[0].id
        for stmt in transformed.body
        if isinstance(stmt, ast.Assign) and isinstance(stmt.targets[0], ast.Name)
    ]
    assert "x" in names


# ---------------------------------------------------------------------------
# 3. _resolve_annotation_aliases
# ---------------------------------------------------------------------------

def test_resolve_annotation_aliases_simple_name():
    """A Name that is a known alias must be expanded to the aliased Subscript."""
    pre = _make_pre()
    pre.type_aliases["Coordinate"] = ast.parse("Tuple[int, int]", mode="eval").body

    annotation = ast.Name(id="Coordinate", ctx=ast.Load())
    resolved = pre._resolve_annotation_aliases(annotation)

    assert _get_subscript_name(resolved) == "Tuple"


def test_resolve_annotation_aliases_unknown_name_unchanged():
    pre = _make_pre()
    annotation = ast.Name(id="int", ctx=ast.Load())
    resolved = pre._resolve_annotation_aliases(annotation)
    assert _get_annotation_name(resolved) == "int"


def test_resolve_annotation_aliases_none_returns_none():
    pre = _make_pre()
    assert pre._resolve_annotation_aliases(None) is None


def test_resolve_annotation_aliases_nested():
    """List[Coordinate] where Coordinate = Tuple[int,int] → List[Tuple[int,int]]"""
    pre = _make_pre()
    pre.type_aliases["Coordinate"] = ast.parse("Tuple[int, int]", mode="eval").body

    annotation = ast.parse("List[Coordinate]", mode="eval").body
    resolved = pre._resolve_annotation_aliases(annotation)

    assert _get_subscript_name(resolved) == "List"
    # The slice of the outer List should now be Tuple[int, int]
    inner = resolved.slice
    # Python 3.9+ stores the slice directly; 3.8 wraps in ast.Index
    if isinstance(inner, ast.Index):
        inner = inner.value
    assert _get_subscript_name(inner) == "Tuple"


def test_resolve_annotation_aliases_tuple_elements():
    """Tuple[Coordinate, int] → Tuple[Tuple[int,int], int]"""
    pre = _make_pre()
    pre.type_aliases["Coordinate"] = ast.parse("Tuple[int, int]", mode="eval").body

    annotation = ast.parse("Tuple[Coordinate, int]", mode="eval").body
    resolved = pre._resolve_annotation_aliases(annotation)

    assert _get_subscript_name(resolved) == "Tuple"
    # First element of the slice tuple should be expanded
    inner = resolved.slice
    if isinstance(inner, ast.Index):
        inner = inner.value
    assert isinstance(inner, ast.Tuple)
    first_elt = inner.elts[0]
    assert _get_subscript_name(first_elt) == "Tuple"


# ---------------------------------------------------------------------------
# 4. visit_AnnAssign – aliases resolved in variable annotations
# ---------------------------------------------------------------------------

def test_annassign_alias_resolved():
    """coord: Coordinate should resolve to coord: Tuple[int, int] after visit."""
    src = (
        "from typing import Tuple\n"
        "Coordinate = Tuple[int, int]\n"
        "coord: Coordinate = (0, 0)\n"
    )
    module = ast.parse(src)
    pre = _make_pre()
    transformed = pre.visit(module)

    # Find the AnnAssign for coord
    ann_assign = next(
        (s for s in transformed.body
         if isinstance(s, ast.AnnAssign) and isinstance(s.target, ast.Name)
         and s.target.id == "coord"),
        None,
    )
    assert ann_assign is not None, "AnnAssign for coord not found"
    assert _get_subscript_name(ann_assign.annotation) == "Tuple", (
        f"Expected Tuple annotation, got {ast.dump(ann_assign.annotation)}"
    )


# ---------------------------------------------------------------------------
# 5. visit_FunctionDef – aliases resolved in return type and param annotations
# ---------------------------------------------------------------------------

def test_function_return_alias_resolved():
    """-> Coordinate should become -> Tuple[int, int]."""
    src = (
        "from typing import Tuple\n"
        "Coordinate = Tuple[int, int]\n"
        "def make(x: int, y: int) -> Coordinate:\n"
        "    return (x, y)\n"
    )
    module = ast.parse(src)
    pre = _make_pre()
    transformed = pre.visit(module)

    func = next(
        (s for s in transformed.body if isinstance(s, ast.FunctionDef) and s.name == "make"),
        None,
    )
    assert func is not None
    assert _get_subscript_name(func.returns) == "Tuple", (
        f"Expected Tuple return annotation, got {ast.dump(func.returns)}"
    )


def test_function_param_alias_resolved():
    """param: Coordinate should become param: Tuple[int, int]."""
    src = (
        "from typing import Tuple\n"
        "Coordinate = Tuple[int, int]\n"
        "def distance(a: Coordinate, b: Coordinate) -> int:\n"
        "    return 0\n"
    )
    module = ast.parse(src)
    pre = _make_pre()
    transformed = pre.visit(module)

    func = next(
        (s for s in transformed.body
         if isinstance(s, ast.FunctionDef) and s.name == "distance"),
        None,
    )
    assert func is not None
    for arg in func.args.args:
        if arg.arg in ("a", "b"):
            assert _get_subscript_name(arg.annotation) == "Tuple", (
                f"Param {arg.arg}: expected Tuple, got {ast.dump(arg.annotation)}"
            )


# ---------------------------------------------------------------------------
# 6. expand_dataclass – __init__ inserted after docstring, not before
# ---------------------------------------------------------------------------

def test_dataclass_init_inserted_after_docstring():
    """When a dataclass has a docstring, __init__ must come after it."""
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Point:\n"
        '    """A 2-D point."""\n'
        "    x: int\n"
        "    y: int\n"
    )
    module = ast.parse(src)
    pre = _make_pre()
    transformed = pre.visit(module)

    cls = next(
        (s for s in transformed.body if isinstance(s, ast.ClassDef) and s.name == "Point"),
        None,
    )
    assert cls is not None, "Class Point not found after transform"

    # First statement must still be the docstring
    first = cls.body[0]
    assert isinstance(first, ast.Expr), "First stmt should be Expr (docstring)"
    assert isinstance(first.value, ast.Constant) and isinstance(first.value.value, str), (
        "First stmt should be a string constant (docstring)"
    )

    # Second statement must be the generated __init__
    second = cls.body[1]
    assert isinstance(second, ast.FunctionDef) and second.name == "__init__", (
        "__init__ must be the second statement after the docstring"
    )


def test_dataclass_init_inserted_at_index_0_without_docstring():
    """Without a docstring, __init__ must be the first statement."""
    src = (
        "from dataclasses import dataclass\n"
        "@dataclass\n"
        "class Point:\n"
        "    x: int\n"
        "    y: int\n"
    )
    module = ast.parse(src)
    pre = _make_pre()
    transformed = pre.visit(module)

    cls = next(
        (s for s in transformed.body if isinstance(s, ast.ClassDef) and s.name == "Point"),
        None,
    )
    assert cls is not None
    first = cls.body[0]
    assert isinstance(first, ast.FunctionDef) and first.name == "__init__", (
        "__init__ must be the first statement when there is no docstring"
    )


# ---------------------------------------------------------------------------
# 7. for-loop tuple unpacking in iterable lowering
# ---------------------------------------------------------------------------

def test_iterable_for_tuple_unpack_inserts_target_assignments():
    """for a, b in xs must define a and b in transformed loop body."""
    src = (
        "from typing import List, Tuple\n"
        "def get_xs() -> List[Tuple[int, int]]:\n"
        "    return [(1, 2)]\n"
        "xs: List[Tuple[int, int]] = get_xs()\n"
        "for a, b in xs:\n"
        "    c = a + b\n"
    )
    module = ast.parse(src)
    pre = _make_pre()
    transformed = pre.visit(module)

    while_node = next((s for s in transformed.body if isinstance(s, ast.While)), None)
    assert while_node is not None, "Expected transformed while-loop"

    assigned_names = []
    for stmt in while_node.body:
        if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
            assigned_names.append(stmt.target.id)
        elif isinstance(stmt, ast.Assign):
            for tgt in stmt.targets:
                if isinstance(tgt, ast.Name):
                    assigned_names.append(tgt.id)

    assert "a" in assigned_names, "Tuple-unpacked variable 'a' must be assigned"
    assert "b" in assigned_names, "Tuple-unpacked variable 'b' must be assigned"


def test_iterable_for_tuple_unpack_before_body_use():
    """Tuple unpack assignments must be emitted before original loop body statements."""
    src = (
        "from typing import List, Tuple\n"
        "def get_xs() -> List[Tuple[int, int]]:\n"
        "    return [(1, 2)]\n"
        "xs: List[Tuple[int, int]] = get_xs()\n"
        "for time, task in xs:\n"
        "    print(time)\n"
    )
    module = ast.parse(src)
    pre = _make_pre()
    transformed = pre.visit(module)

    while_node = next((s for s in transformed.body if isinstance(s, ast.While)), None)
    assert while_node is not None

    # Locate first usage of 'time' in loop body and ensure an assignment to
    # 'time' appears before that statement.
    first_time_use_idx = None
    for i, stmt in enumerate(while_node.body):
        if isinstance(stmt, ast.Expr):
            has_time = any(isinstance(n, ast.Name) and n.id == "time" for n in ast.walk(stmt))
            if has_time:
                first_time_use_idx = i
                break

    assert first_time_use_idx is not None, "Expected a statement using 'time'"

    has_prior_time_assign = any(
        (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.target.id == "time"
        )
        or (
            isinstance(stmt, ast.Assign)
            and any(isinstance(tgt, ast.Name) and tgt.id == "time" for tgt in stmt.targets)
        )
        for stmt in while_node.body[:first_time_use_idx]
    )
    assert has_prior_time_assign, "'time' must be assigned before first use"


# ---------------------------------------------------------------------------
# 8. for-loop unrolling over tracked list literals
# ---------------------------------------------------------------------------

def test_for_over_list_literal_var_is_unrolled():
    src = (
        "xs = [1, 2, 3]\n"
        "total = 0\n"
        "for x in xs:\n"
        "    total = total + x\n"
    )
    module = ast.parse(src)
    pre = _make_pre()
    transformed = pre.visit(module)

    # Unrolled form should not contain a while-loop for this case.
    while_nodes = [n for n in transformed.body if isinstance(n, ast.While)]
    assert len(while_nodes) == 0

    # Expect direct assignments to x for each literal element.
    x_assigns = [
        s for s in transformed.body
        if (
            isinstance(s, ast.AnnAssign)
            and isinstance(s.target, ast.Name)
            and s.target.id == "x"
        )
        or (
            isinstance(s, ast.Assign)
            and any(isinstance(tgt, ast.Name) and tgt.id == "x" for tgt in s.targets)
        )
    ]
    assert len(x_assigns) == 3


def test_for_tuple_unpack_over_list_literal_var_is_unrolled():
    src = (
        "pairs = [(1, 10), (2, 20)]\n"
        "for a, b in pairs:\n"
        "    c = a + b\n"
    )
    module = ast.parse(src)
    pre = _make_pre()
    transformed = pre.visit(module)

    while_nodes = [n for n in transformed.body if isinstance(n, ast.While)]
    assert len(while_nodes) == 0

    tuple_unpack_assigns = [
        s
        for s in transformed.body
        if isinstance(s, ast.Assign)
        and len(s.targets) == 1
        and isinstance(s.targets[0], ast.Tuple)
        and len(s.targets[0].elts) == 2
        and all(isinstance(e, ast.Name) for e in s.targets[0].elts)
        and [e.id for e in s.targets[0].elts] == ["a", "b"]
    ]
    assert len(tuple_unpack_assigns) == 2
