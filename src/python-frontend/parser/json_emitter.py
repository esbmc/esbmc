"""AST JSON emission helpers for parser pipeline."""

from __future__ import annotations

import ast
import json
import os
from dataclasses import dataclass
from typing import Callable, Iterable, Protocol

from libs.ast2json import ast2json as ast2json_func

__all__ = [
    "EmitPipelineDeps",
    "JsonEmitterDeps",
    "emit_file_as_json",
    "emit_module_json",
    "generate_ast_json",
]


class ImportResolverLike(Protocol):
    """Structural type for resolver functions consumed by JSON emission."""
    # pylint: disable=missing-function-docstring

    def filter_imports(self, tree: ast.Module) -> ast.Module:
        ...

    def resolve_module_file(self, module_qualname: str, output_dir: str) -> str | None:
        ...


@dataclass(frozen=True)
class JsonEmitterDeps:
    """Dependencies required by AST->JSON emission."""
    import_resolver: ImportResolverLike
    tag_bignum_constants: Callable[[object], None]


@dataclass(frozen=True)
class EmitPipelineDeps:
    """Dependencies for file/module emission wrappers."""
    parse_file: Callable[[str], tuple[ast.AST, object]] | None = None
    generate_ast_json_fn: Callable[..., None] | None = None
    import_resolver: ImportResolverLike | None = None
    emit_file_as_json_fn: Callable[..., None] | None = None


def get_referenced_names(node: ast.AST) -> set[str]:
    """Find function/class names referenced in a function or class definition."""
    referenced = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            if isinstance(child.func, ast.Name):
                referenced.add(child.func.id)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if child.returns and isinstance(child.returns, ast.Name):
                referenced.add(child.returns.id)
            for arg in child.args.args:
                if arg.annotation and isinstance(arg.annotation, ast.Name):
                    referenced.add(arg.annotation.id)
        elif isinstance(child, ast.AnnAssign):
            if isinstance(child.annotation, ast.Name):
                referenced.add(child.annotation.id)

    return referenced


def _assign_target_names(target: ast.expr) -> Iterable[str]:
    if isinstance(target, ast.Name):
        yield target.id
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            yield from _assign_target_names(elt)
    elif isinstance(target, ast.Starred):
        yield from _assign_target_names(target.value)


def _filter_nodes_for_import(tree: ast.Module, elements_to_import) -> list[ast.stmt]:
    """Return the subset of ``tree.body`` selected by ``elements_to_import``."""
    if not elements_to_import:
        return []

    explicitly_imported = {elem_info.name for elem_info in elements_to_import}
    referenced_names = set()
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name in explicitly_imported:
            referenced_names.update(get_referenced_names(node))

    filtered_nodes = []
    for node in tree.body:
        if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
            if (node.name in ("ESBMC_range_has_next_", "ESBMC_range_next_")
                    or node.name in explicitly_imported or node.name in referenced_names):
                filtered_nodes.append(node)
        elif (isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
              and node.target.id in explicitly_imported):
            filtered_nodes.append(node)
        elif isinstance(node, ast.Assign):
            bound = {n for tgt in node.targets for n in _assign_target_names(tgt)}
            if bound & explicitly_imported:
                filtered_nodes.append(node)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            filtered_nodes.append(node)
    return filtered_nodes


def _compute_output_json_path(
    python_filename: str,
    output_dir: str,
    module_qualname: str | None,
) -> str:
    if module_qualname:
        parts = module_qualname.split(".")
        return os.path.join(output_dir, *parts[:-1], f"{parts[-1]}.json")
    if python_filename.endswith('__init__.py'):
        dir_name = os.path.basename(os.path.dirname(python_filename))
        return os.path.join(output_dir, f"{dir_name}.json")
    return os.path.join(output_dir, f"{os.path.basename(python_filename[:-3])}.json")


# pylint: disable-next=too-many-arguments
def generate_ast_json(
    tree: ast.Module,
    python_filename: str,
    elements_to_import,
    output_dir: str,
    module_qualname=None,
    *,
    deps: JsonEmitterDeps,
):
    """Generate AST JSON from the given Python AST tree."""
    tree = deps.import_resolver.filter_imports(tree)
    filtered_nodes = _filter_nodes_for_import(tree, elements_to_import)

    ast_json = ast2json_func(
        ast.Module(body=filtered_nodes, type_ignores=[]) if filtered_nodes else tree)
    ast_json["filename"] = python_filename
    ast_json["ast_output_dir"] = output_dir
    deps.tag_bignum_constants(ast_json)

    json_filename = _compute_output_json_path(python_filename, output_dir, module_qualname)
    os.makedirs(os.path.dirname(json_filename), exist_ok=True)

    with open(json_filename, "w", encoding="utf-8") as json_file:
        json.dump(ast_json, json_file, indent=4, ensure_ascii=False)


def emit_file_as_json(
    filename: str,
    output_dir: str,
    module_qualname: str | None = None,
    elements_to_import=None,
    *,
    deps: EmitPipelineDeps,
) -> None:
    """Generate AST JSON for a source file."""
    if deps.parse_file is None or deps.generate_ast_json_fn is None:
        raise ValueError("EmitPipelineDeps requires parse_file and generate_ast_json_fn")
    tree, _ = deps.parse_file(filename)
    deps.generate_ast_json_fn(
        tree,
        filename,
        elements_to_import,
        output_dir,
        module_qualname=module_qualname,
    )


def emit_module_json(
    module_qualname: str,
    output_dir: str,
    elements_to_import=None,
    *,
    deps: EmitPipelineDeps,
) -> None:
    """Resolve module to file and emit AST JSON."""
    if deps.import_resolver is None or deps.emit_file_as_json_fn is None:
        raise ValueError("EmitPipelineDeps requires import_resolver and emit_file_as_json_fn")
    filename = deps.import_resolver.resolve_module_file(module_qualname, output_dir)
    if filename:
        deps.emit_file_as_json_fn(filename, output_dir, module_qualname, elements_to_import)
