from __future__ import annotations

import ast
import importlib
import os
import sys
from dataclasses import dataclass
from types import ModuleType
from typing import Any, Callable, TypeAlias, TypedDict


class ModuleImportInfo(TypedDict):
    import_all: bool
    specific_names: set[str]


ParseFileCanonicalisedFn: TypeAlias = Callable[[str], tuple[ast.AST, Any]]
RewriteRelativeImportFn: TypeAlias = Callable[[ast.ImportFrom, str | None], None]
SnapshotExportsFn: TypeAlias = Callable[[Any], tuple[set[str], dict[str, str], list[str] | None]]
PropagateRangeAliasesFn: TypeAlias = Callable[[dict[str, tuple[ast.AST, str, Any]]], None]
GenerateAstJsonFn: TypeAlias = Callable[[ast.AST, str, Any, str, str | None], None]

import_aliases: dict[str, str] = {}
module_imports: dict[str, ModuleImportInfo] = {}
module_exports: dict[str, tuple[set[str], dict[str, str], list[str] | None]] = {}
_reported_cycles: set[tuple[str, ...]] = set()
_reported_resolution_failures: set[tuple[str, str, str]] = set()


@dataclass(frozen=True)
class ResolverCallbacks:
    parse_file_canonicalised: ParseFileCanonicalisedFn
    rewrite_relative_import: RewriteRelativeImportFn
    snapshot_exports: SnapshotExportsFn
    propagate_range_aliases: PropagateRangeAliasesFn
    generate_ast_json: GenerateAstJsonFn


def reset_state() -> None:
    """Reset resolver global state for a fresh parser run."""
    import_aliases.clear()
    module_imports.clear()
    module_exports.clear()
    _reported_cycles.clear()
    _reported_resolution_failures.clear()


def _resolver_error(module_name: str, reason: str, detail: str | None = None) -> None:
    message = f"ERROR: import resolver [{reason}] for '{module_name}'"
    if detail:
        message = f"{message}: {detail}"
    print(message)


def _resolver_warning(module_name: str, reason: str, detail: str | None = None) -> None:
    message = f"WARNING: import resolver [{reason}] for '{module_name}'"
    if detail:
        message = f"{message}: {detail}"
    print(message)


def _node_location(node: ast.AST) -> str:
    line = getattr(node, "lineno", None)
    col = getattr(node, "col_offset", None)
    if line is None or col is None:
        return "?:?"
    return f"{line}:{col}"


def _mark_import_resolution(
    node: ast.AST,
    ok: bool,
    full_path: str | None = None,
    reason: str | None = None,
) -> None:
    node.full_path = full_path
    node.module_not_found = not ok
    node.module_resolution_reason = reason


def _warn_resolution_failure(module_name: str, node: ast.AST, reason: str) -> None:
    location = _node_location(node)
    key = (module_name, reason, location)
    if key in _reported_resolution_failures:
        return
    _reported_resolution_failures.add(key)
    _resolver_warning(module_name, "resolution-failed", f"{reason} at {location}")


def _is_imported_model(module_name: str) -> bool:
    models = {
        "math",
        "cmath",
        "os",
        "numpy",
        "esbmc",
        "decimal",
        "collections",
        "dataclasses",
        "typing",
        "time",
        "threading",
    }
    return module_name in models


def _is_unsupported_module(module_name: str) -> bool:
    return module_name in {"blah"}


def _is_testing_framework(module_name: str) -> bool:
    return module_name in {"pytest"}


def _base_module_name(module_name: str) -> str:
    return module_name.split(".")[0]


def _is_standard_library_file(filename: str) -> bool:
    stdlib_paths = [
        '/usr/lib/python',
        '/usr/local/lib/python',
        '/Library/Frameworks/Python.framework',
        '/opt/homebrew/Cellar/python',
        '/usr/local/Cellar/python',
        '/opt/conda/lib/python',
    ]
    if any(filename.startswith(path) for path in stdlib_paths):
        return True

    pyenv_root = os.environ.get('PYENV_ROOT', os.path.expanduser('~/.pyenv'))
    if pyenv_root and filename.startswith(pyenv_root):
        if '/versions/' in filename and '/lib/python' in filename:
            return True

    if filename.startswith(os.path.expanduser('~/miniconda3/lib/python')) or \
       filename.startswith(os.path.expanduser('~/anaconda3/lib/python')):
        return True

    return False


def _normalize_existing_path(path: str | None) -> str | None:
    if not path:
        return None
    absolute = os.path.abspath(path)
    if not os.path.exists(absolute):
        return None
    return absolute


def _module_filename(module: ModuleType | str | None) -> str | None:
    candidate = module if isinstance(module, str) else getattr(module, "__file__", None)
    normalized = _normalize_existing_path(candidate)
    if not normalized:
        return None
    if _is_standard_library_file(normalized):
        return None
    return normalized


def import_module_by_name(
    module_name: str,
    output_dir: str,
) -> ModuleType | str | None:
    if _is_unsupported_module(module_name):
        _resolver_error(
            module_name,
            "unsupported-module",
            f"\"import {module_name}\" is not supported",
        )
        sys.exit(3)

    base_module = _base_module_name(module_name)

    if _is_testing_framework(base_module):
        return None
    if _is_imported_model(base_module):
        parts = module_name.split(".")
        model_dir = os.path.join(output_dir, "models")
        path = os.path.join(model_dir, *parts) + ".py"

        if not os.path.exists(path):
            path = os.path.join(model_dir, *parts, "__init__.py")

        normalized = _normalize_existing_path(path)
        if normalized is None:
            _resolver_error(module_name, "model-not-found", f"expected model file at '{path}'")
            return None
        return normalized

    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        if "." in module_name:
            parent = ".".join(module_name.split(".")[:-1])
            try:
                return importlib.import_module(parent)
            except ImportError:
                pass

        _resolver_error(module_name, "module-not-found", "install it with: pip3 install " + module_name)
        return None


def expand_star_import(module) -> list[str] | None:
    names = getattr(module, '__all__', None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith('_')]
    return names


def _collect_import_targets(
    node: ast.Import | ast.ImportFrom, ) -> tuple[list[str], list[ast.alias] | None]:
    if isinstance(node, ast.Import):
        module_names: list[str] = []
        for alias_node in node.names:
            module_name = alias_node.name
            alias = alias_node.asname or module_name
            import_aliases[alias] = module_name
            module_names.append(module_name)
        return module_names, None

    module_name = node.module
    if module_name:
        import_aliases[module_name] = module_name
    module_names = [module_name] if module_name else []
    imported_elements = None if any(a.name == '*' for a in node.names) else node.names
    return module_names, imported_elements


def process_imports(node: ast.Import | ast.ImportFrom, output_dir: str) -> None:
    module_names, imported_elements = _collect_import_targets(node)
    if not module_names:
        return

    for module_name in module_names:
        if module_name not in module_imports:
            module_imports[module_name] = {'import_all': False, 'specific_names': set()}

        if imported_elements is None:
            module_imports[module_name]['import_all'] = True
        else:
            for elem in imported_elements:
                module_imports[module_name]['specific_names'].add(elem.name)

        module = import_module_by_name(module_name, output_dir)
        if module is None:
            _warn_resolution_failure(module_name, node, "module-not-found")
            _mark_import_resolution(node, ok=False, reason="module-not-found")
            continue
        filename = _module_filename(module)
        if filename is None:
            _warn_resolution_failure(module_name, node, "module-file-not-found")
            _mark_import_resolution(node, ok=False, reason="module-file-not-found")
            continue
        _mark_import_resolution(node, ok=True, full_path=filename)


def resolve_module_file(module_qualname: str, output_dir: str) -> str | None:
    try:
        mod = import_module_by_name(module_qualname, output_dir)
    except SystemExit:
        return None
    filename = _module_filename(mod)
    if filename is None:
        return None
    return filename


def _extract_module_names(node: ast.Import | ast.ImportFrom) -> list[str]:
    names: list[str]
    if isinstance(node, ast.Import):
        names = [alias_node.name for alias_node in node.names]
    else:
        names = [node.module] if node.module else []

    filtered: list[str] = []
    for name in names:
        if not name:
            continue
        if _is_testing_framework(_base_module_name(name)):
            continue
        filtered.append(name)
    return filtered


def _find_cycle_from(graph: dict[str, set[str]], start: str) -> tuple[str, ...] | None:
    stack: list[str] = [start]
    in_stack: set[str] = {start}

    def dfs(current: str) -> tuple[str, ...] | None:
        for nxt in graph.get(current, set()):
            if nxt in in_stack:
                idx = stack.index(nxt)
                return tuple(stack[idx:] + [nxt])
            if nxt not in graph:
                continue
            stack.append(nxt)
            in_stack.add(nxt)
            found = dfs(nxt)
            if found:
                return found
            in_stack.remove(nxt)
            stack.pop()
        return None

    return dfs(start)


def _warn_if_cycle(import_graph: dict[str, set[str]], module_name: str) -> None:
    cycle = _find_cycle_from(import_graph, module_name)
    if not cycle or cycle in _reported_cycles:
        return
    _reported_cycles.add(cycle)
    readable = " -> ".join(cycle)
    _resolver_warning(module_name, "cyclic-import", readable)


def filter_imports(tree: ast.AST) -> ast.AST:
    filtered_body = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            filtered_names = []
            for alias in node.names:
                base_module = _base_module_name(alias.name)
                if not _is_testing_framework(base_module):
                    filtered_names.append(alias)
            if filtered_names:
                node.names = filtered_names
                filtered_body.append(node)

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                base_module = _base_module_name(node.module)
                if not _is_testing_framework(base_module):
                    filtered_body.append(node)
            else:
                filtered_body.append(node)
        else:
            filtered_body.append(node)

    tree.body = filtered_body
    return tree


def process_collected_imports(output_dir: str, callbacks: ResolverCallbacks):
    """Emit collected imports after transitive discovery converges.

    Callback parameters wire parser-owned routines (parse, rewrite propagation,
    and AST emission) without coupling this module to parser internals.
    """
    parsed_trees = {}
    visited = set()
    import_graph: dict[str, set[str]] = {}

    while True:
        pending = sorted(set(module_imports.keys()) - visited)
        if not pending:
            break
        for module_name in pending:
            visited.add(module_name)
            filename = resolve_module_file(module_name, output_dir)
            if not filename:
                _resolver_warning(module_name, "module-file-not-found")
                continue
            try:
                tree, preprocessor = callbacks.parse_file_canonicalised(filename)
            except (OSError, SyntaxError, UnicodeDecodeError) as exc:
                _resolver_warning(module_name, "module-parse-failed", str(exc))
                continue
            parsed_trees[module_name] = (tree, filename, preprocessor)
            module_exports[module_name] = callbacks.snapshot_exports(preprocessor)
            import_graph.setdefault(module_name, set())
            for subnode in ast.walk(tree):
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                    callbacks.rewrite_relative_import(subnode, module_name)
                    for imported_name in _extract_module_names(subnode):
                        import_graph[module_name].add(imported_name)
                    process_imports(subnode, output_dir)
            _warn_if_cycle(import_graph, module_name)

    callbacks.propagate_range_aliases(parsed_trees)

    for _module_name, (tree, _filename, preprocessor) in parsed_trees.items():
        preprocessor.finalize_module(tree)

    _emit_collected_import_json(parsed_trees, output_dir, callbacks)


def _emit_collected_import_json(
    parsed_trees: dict[str, tuple[ast.AST, str, Any]],
    output_dir: str,
    callbacks: ResolverCallbacks,
) -> None:
    for module_name, import_info in module_imports.items():
        imported_elements = None if import_info['import_all'] else [
            ast.alias(name, None) for name in import_info['specific_names']
        ]

        for name in sorted(import_info['specific_names']):
            emit_module_json(
                f"{module_name}.{name}",
                output_dir,
                callbacks.generate_ast_json,
                callbacks.parse_file_canonicalised,
            )

        if module_name not in parsed_trees:
            continue

        tree, filename, _preprocessor = parsed_trees[module_name]
        callbacks.generate_ast_json(
            tree,
            filename,
            imported_elements,
            output_dir,
            module_qualname=module_name,
        )


def rewrite_relative_import(node: ast.ImportFrom, parent_module: str | None) -> None:
    lvl = getattr(node, "level", 0)
    if lvl <= 0 or not parent_module:
        return

    parts = parent_module.split(".")
    if lvl > len(parts):
        _resolver_warning(
            parent_module,
            "relative-import-level-too-high",
            f"level={lvl}, target={node.module or '<package>'}",
        )
    idx = len(parts) - lvl
    base = parent_module if idx <= 0 else ".".join(parts[:idx])
    node.module = f"{base}.{node.module}" if node.module else base
    node.level = 0


def _emit_submodule_asts(
    module_dir: str,
    base_module: str,
    output_dir: str,
    generate_ast_json_fn: GenerateAstJsonFn,
) -> None:
    for root, dirs, files in os.walk(module_dir):
        # Keep traversal deterministic across platforms/runs for stable CI output.
        dirs.sort()
        for file in sorted(files):
            if not file.endswith('.py'):
                continue
            full_path = os.path.join(root, file)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
            except (OSError, UnicodeDecodeError, SyntaxError) as exc:
                _resolver_warning(
                    f"{base_module}.{file}",
                    "submodule-parse-failed",
                    f"{full_path}: {exc}",
                )
                continue
            generate_ast_json_fn(tree, full_path, None, f"{output_dir}/{base_module}")


def detect_and_process_submodules(
    node: ast.AST,
    processed_submodules: set[str],
    output_dir: str,
    generate_ast_json_fn: GenerateAstJsonFn,
) -> None:
    if not isinstance(node, ast.Attribute):
        return

    value = node.value
    if not isinstance(value, ast.Name):
        return

    base_module = import_aliases.get(value.id)
    if not base_module or not _is_imported_model(base_module):
        return

    full_module = f"{base_module}.{node.attr}"
    if full_module in processed_submodules:
        return

    processed_submodules.add(full_module)
    try:
        module = import_module_by_name(full_module, output_dir)
    except SystemExit:
        return

    file_path = _module_filename(module)
    if not file_path:
        return

    _emit_submodule_asts(
        os.path.dirname(file_path),
        base_module,
        output_dir,
        generate_ast_json_fn,
    )


def emit_module_json(
    module_qualname: str,
    output_dir: str,
    generate_ast_json_fn: GenerateAstJsonFn,
    parse_file_canonicalised_fn: ParseFileCanonicalisedFn,
) -> None:
    filename = resolve_module_file(module_qualname, output_dir)
    if not filename:
        _resolver_warning(module_qualname, "module-file-not-found")
        return
    try:
        tree, _preprocessor = parse_file_canonicalised_fn(filename)
    except (OSError, SyntaxError, UnicodeDecodeError) as exc:
        _resolver_warning(module_qualname, "module-parse-failed", str(exc))
        return
    generate_ast_json_fn(tree, filename, None, output_dir, module_qualname=module_qualname)
