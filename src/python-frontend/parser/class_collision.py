"""Separate an imported module's class from a same-named program class.

ESBMC keys a class symbol by its name alone, so a class in the program file and
a same-named class in an imported module would share one symbol (#7397). The
imported module's class is renamed to ``<name>$<module>`` throughout that module
and at every reference to it from an importing module, before any JSON is
emitted. A collision is renamed only in the shapes whose scoping is plain;
any other shape keeps the shared name and the converter refuses it.
"""

from __future__ import annotations

import ast
from typing import Iterable, Iterator

# Nodes that bind a plain-string `name` rather than through an ast.Name.
_NAMED_BINDERS = (ast.ExceptHandler, ast.MatchAs, ast.MatchStar) + tuple(
    getattr(ast, n) for n in ("TypeVar", "ParamSpec", "TypeVarTuple") if hasattr(ast, n))


def top_level_classes(tree: ast.Module) -> set[str]:
    """Names of the classes defined directly in a module's body."""
    return {node.name for node in tree.body if isinstance(node, ast.ClassDef)}


def qualified_name(name: str, module_name: str) -> str:
    """The name an imported module's colliding class is converted under."""
    return f"{name}${module_name.replace('.', '$')}"


def _dotted(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = _dotted(node.value)
        return None if base is None else f"{base}.{node.attr}"
    return None


def _module_aliases(tree: ast.AST, module_name: str) -> set[str]:
    return {
        alias.asname or alias.name
        for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names
        if alias.name == module_name
    }


def _from_imports(tree: ast.AST, module_name: str) -> Iterator[ast.ImportFrom]:
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module == module_name and node.level == 0:
            yield node


def _annotations(tree: ast.AST) -> Iterator[ast.AST]:
    for node in ast.walk(tree):
        if isinstance(node, ast.arg) and node.annotation is not None:
            yield node.annotation
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.returns:
            yield node.returns
        elif isinstance(node, ast.AnnAssign):
            yield node.annotation


def _import_binds(node: ast.AST, name: str, allowed_module: str | None) -> bool:
    if isinstance(node, ast.Import):
        return any((a.asname or a.name) == name for a in node.names)
    if isinstance(node, ast.ImportFrom) and node.module != allowed_module:
        return any((a.asname or a.name) in (name, "*") for a in node.names)
    return False


def _non_class_binds(node: ast.AST, name: str) -> bool:
    if isinstance(node, ast.Name):
        return node.id == name and not isinstance(node.ctx, ast.Load)
    if isinstance(node, ast.arg):
        return node.arg == name
    if isinstance(node, (ast.Global, ast.Nonlocal)):
        return name in node.names
    if isinstance(node, ast.MatchMapping):
        return node.rest == name
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef) + _NAMED_BINDERS):
        return getattr(node, "name", None) == name
    return False


def _binds_otherwise(tree: ast.Module,
                     name: str,
                     own_class: bool,
                     allowed_module: str | None = None) -> bool:
    """Whether ``tree`` binds ``name`` other than by a plain read.

    ``own_class`` allows the one top-level ``class name``; imports from
    ``allowed_module`` are judged by the caller.
    """

    def binds(node: ast.AST) -> bool:
        if isinstance(node, ast.ClassDef):
            return node.name == name and not (own_class and node in tree.body)
        return _non_class_binds(node, name) or _import_binds(node, name, allowed_module)

    return any(binds(node) for node in ast.walk(tree))


def _reads(nodes: Iterable[ast.AST], name: str) -> bool:
    """Whether ``name`` is read anywhere in ``nodes``, function bodies included:
    a function defined before the class may run before it too."""
    return any(isinstance(n, ast.Name) and n.id == name for node in nodes for n in ast.walk(node))


def _importer_is_plain(tree: ast.Module, module_name: str, name: str) -> bool:
    """Whether ``tree``'s references to ``module_name``'s ``name`` can be renamed."""
    imports = list(_from_imports(tree, module_name))
    if any(alias.name == "*" for node in imports for alias in node.names):
        return False
    # `from m import name as other` binds `other`, which no local class shadows.
    named = [
        node for node in imports
        if any(alias.name == name and alias.asname is None for alias in node.names)
    ]
    if not named:
        return True
    if name not in top_level_classes(tree):
        return not _binds_otherwise(tree, name, own_class=False, allowed_module=module_name)
    return _own_class_shadows(tree, name, named)


def _own_class_shadows(tree: ast.Module, name: str, named: list[ast.ImportFrom]) -> bool:
    """The importer's own class shadows the import only if every import of the
    name runs, at module level, before the class and nothing reads it between."""
    class_index, own = next((i, node) for i, node in enumerate(tree.body)
                            if isinstance(node, ast.ClassDef) and node.name == name)
    # The class header runs before the class name is bound.
    header = own.bases + own.keywords + own.decorator_list
    for node in named:
        if node not in tree.body or tree.body.index(node) > class_index:
            return False
        if _reads(tree.body[tree.body.index(node) + 1:class_index] + header, name):
            return False
    return True


def _rename_references(tree: ast.AST, renames: dict[str, str]) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name in renames:
            node.name = renames[node.name]
        elif isinstance(node, ast.Name) and node.id in renames:
            node.id = renames[node.id]
    # A string names a type only inside an annotation; elsewhere it is data.
    for annotation in _annotations(tree):
        for node in ast.walk(annotation):
            if isinstance(node, ast.Constant) and node.value in renames:
                node.value = renames[node.value]


def _rewrite_importer(tree: ast.Module, module_name: str, name: str, new: str) -> None:
    aliases = _module_aliases(tree, module_name)
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == name and _dotted(node.value) in aliases:
            node.attr = new
    from_imported = False
    for node in _from_imports(tree, module_name):
        for alias in node.names:
            if alias.name == name:
                alias.name = new
                from_imported = from_imported or alias.asname is None
    if from_imported and name not in top_level_classes(tree):
        _rename_references(tree, {name: new})


def separate_colliding_classes(entry_tree: ast.Module, trees: dict[str, ast.Module],
                               skip: Iterable[str]) -> dict[str, dict[str, str]]:
    """Rename each imported module's classes that the program file also defines.

    ``trees`` maps module name to AST; modules in ``skip`` (operational models)
    are left alone. Returns the renames applied, per module.
    """
    program_classes = top_level_classes(entry_tree)
    skipped = set(skip)
    applied: dict[str, dict[str, str]] = {}
    for module_name, tree in trees.items():
        if module_name in skipped:
            continue
        importers = [entry_tree] + [t for n, t in trees.items() if n != module_name]
        for name in sorted(top_level_classes(tree) & program_classes):
            if _binds_otherwise(tree, name, own_class=True) or not all(
                    _importer_is_plain(t, module_name, name) for t in importers):
                continue
            new = qualified_name(name, module_name)
            _rename_references(tree, {name: new})
            for importer in importers:
                _rewrite_importer(importer, module_name, name, new)
            applied.setdefault(module_name, {})[name] = new
    return applied
