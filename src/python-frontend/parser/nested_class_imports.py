"""Name an imported module's nested class by the name it was hoisted to.

The preprocessor lifts a class nested in a module-level class to module level
as ``Outer__Inner`` and rewrites ``Outer.Inner`` within that module (#8256).
An importing module still reads ``mod.Outer.Inner``, or ``Outer.Inner`` after
``from mod import Outer``, so those reads are rewritten here too (#9047). Only
names bound by nothing but the import are rewritten, and only classes that no
module stores to through an attribute or names otherwise.
"""

from __future__ import annotations

import ast
from typing import Iterable

try:
    # pylint: disable=deprecated-module  # `parser` also names a removed stdlib module
    from .class_collision import _dotted, _non_class_binds
except ImportError:
    from class_collision import _dotted, _non_class_binds


def _binders(tree: ast.AST, name: str) -> list[ast.AST]:
    """The nodes of ``tree`` that bind ``name``."""
    found = []
    for node in ast.walk(tree):
        if isinstance(node, ast.alias):
            if node.name != "*" and (node.asname or node.name.split(".")[0]) == name:
                found.append(node)
        elif isinstance(node, ast.ClassDef):
            if node.name == name:
                found.append(node)
        elif _non_class_binds(node, name):
            found.append(node)
    return found


def _module_paths(tree: ast.Module, module_name: str) -> set[str]:
    """The dotted names through which ``tree`` reaches ``module_name`` by
    ``import``, when nothing else in ``tree`` binds their first part."""
    imports = {
        id(alias)
        for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
    }
    paths = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Import):
            continue
        for alias in node.names:
            if alias.name != module_name:
                continue
            path = alias.asname or alias.name
            root = path.split(".")[0]
            if all(
                    id(b) in imports and (
                        b.asname == root and b.name == module_name or b.asname is None
                        and b.name.split(".")[0] == root and module_name.split(".")[0] == root)
                    for b in _binders(tree, root)):
                paths.add(path)
    return paths


def _from_imported(tree: ast.Module, module_name: str,
                   outers: set[str]) -> dict[str, tuple[str, ast.ImportFrom]]:
    """The names ``tree`` binds to a class in ``outers`` by
    ``from module_name import``, and only so."""
    bound = {}
    for node in ast.walk(tree):
        if not (isinstance(node, ast.ImportFrom) and node.module == module_name
                and node.level == 0):
            continue
        for alias in node.names:
            name = alias.asname or alias.name
            if alias.name in outers and _binders(tree, name) == [alias]:
                bound[name] = (alias.name, node)
    return bound


class _QualifyImportedNestedClasses(ast.NodeTransformer):

    def __init__(self, renames, paths, from_imported):
        self.renames = renames
        self.paths = paths
        self.names = {name: (outer, node) for name, (outer, node) in from_imported.items()}
        self.added = {}

    def visit_Attribute(self, node):  # pylint: disable=invalid-name,missing-function-docstring
        self.generic_visit(node)
        value = node.value
        if isinstance(value, ast.Attribute) and _dotted(value.value) in self.paths:
            hoisted = self.renames.get((value.attr, node.attr))
            if hoisted:
                return ast.copy_location(
                    ast.Attribute(value=value.value, attr=hoisted, ctx=ast.Load()), node)
        if isinstance(value, ast.Name) and value.id in self.names:
            outer, import_from = self.names[value.id]
            hoisted = self.renames.get((outer, node.attr))
            if hoisted:
                self.added.setdefault(import_from, set()).add(hoisted)
                self.names[hoisted] = (hoisted, import_from)
                return ast.copy_location(ast.Name(id=hoisted, ctx=ast.Load()), node)
        return node


def _has_star_import(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.ImportFrom) and any(a.name == "*" for a in node.names)
        for node in ast.walk(tree))


def _identifiers(tree: ast.AST) -> set[str]:
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.alias):
            names.add(node.asname or node.name.split(".")[0])
        elif isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)):
            names.add(node.name)
        elif isinstance(node, ast.arg):
            names.add(node.arg)
    return names


def _safe_renames(hoisted: dict[str, dict], importers: list[tuple[str | None,
                                                                  ast.Module]]) -> dict[str, dict]:
    """The renames in ``hoisted`` whose classes no module stores to through
    an attribute, and whose hoisted name no other module mentions: that name
    could resolve to the other module's binding, since ESBMC keys a class by
    its name alone."""
    nodes = [node for _name, tree in importers for node in ast.walk(tree)]
    if any(isinstance(node, ast.Name) and node.id in ("setattr", "delattr") for node in nodes):
        return {}
    stored = {
        node.attr
        for node in nodes if isinstance(node, ast.Attribute) and not isinstance(node.ctx, ast.Load)
    }
    names = {name: _identifiers(tree) for name, tree in importers}
    return {
        module_name: {
            (outer, inner): new
            for (outer, inner), new in renames.items()
            if not {outer, inner} & stored and not any(new in ids for name, ids in names.items()
                                                       if name != module_name)
        }
        for module_name, renames in hoisted.items()
    }


def qualify_imported_nested_classes(hoisted: dict[str, dict],
                                    importers: Iterable[tuple[str | None, ast.Module]]) -> dict:
    """Rewrite each importer's reads of a class hoisted out of a module;
    ``hoisted`` maps a module to its ``(outer, inner) -> name`` renames.
    Returns the hoisted names each module must now export, per module.
    """
    exported: dict[str, set[str]] = {}
    importers = list(importers)
    for module_name, renames in _safe_renames(hoisted, importers).items():
        if not renames:
            continue
        outers = {outer for outer, _inner in renames}
        for importer_name, tree in importers:
            if importer_name == module_name or _has_star_import(tree):
                continue
            paths = _module_paths(tree, module_name)
            from_imported = _from_imported(tree, module_name, outers)
            if not paths and not from_imported:
                continue
            rewriter = _QualifyImportedNestedClasses(renames, paths, from_imported)
            rewriter.visit(tree)
            for node, added in rewriter.added.items():
                node.names.extend(ast.alias(name=n, asname=None) for n in sorted(added))
                ast.fix_missing_locations(node)
                exported.setdefault(module_name, set()).update(added)
    return exported
