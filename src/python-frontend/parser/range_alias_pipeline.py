"""Cross-module range alias propagation helpers."""

from __future__ import annotations

import ast

from preprocessor import Preprocessor

__all__ = [
    "parse_file_canonicalised",
    "snapshot_exports",
    "compute_range_seed",
    "propagate_range_aliases_across_modules",
]


def parse_file_canonicalised(filename: str) -> tuple[ast.AST, Preprocessor]:
    """Parse a file and run only the alias-canonicalisation pre-pass."""
    with open(filename, "r", encoding="utf-8") as src:
        tree = ast.parse(src.read())
    preprocessor = Preprocessor(filename)
    preprocessor.prepare_module(tree)
    return tree, preprocessor


def _snapshot_exports(
    preprocessor: Preprocessor, ) -> tuple[set[str], dict[str, str], set[str] | None]:
    """Snapshot a Preprocessor's range-alias / wrapper export tables."""
    return (
        set(preprocessor.exported_range_aliases),
        dict(preprocessor.exported_range_wrappers),
        preprocessor.module_dunder_all,
    )


def _compute_range_seed(
    module_node: ast.Module,
    import_resolver,
) -> tuple[set[str], dict[str, str]]:
    """Build a (alias_seed, wrapper_seed) pair for *module_node*."""
    alias_seed = set()
    wrapper_seed = {}
    for stmt in module_node.body:
        if not (isinstance(stmt, ast.ImportFrom) and stmt.module):
            continue
        src_exports = import_resolver.module_exports.get(stmt.module)
        if not src_exports:
            continue
        src_aliases, src_wrappers, src_all = src_exports
        if any(a.name == '*' for a in stmt.names):
            visible = (set(src_all) if src_all is not None else
                       {n
                        for n in (set(src_aliases) | set(src_wrappers)) if not n.startswith('_')})
            alias_seed |= (set(src_aliases) & visible)
            for w in src_wrappers:
                if w in visible:
                    wrapper_seed[w] = src_wrappers[w]
            continue
        for a in stmt.names:
            bind_name = a.asname or a.name
            if a.name in src_aliases:
                alias_seed.add(bind_name)
            if a.name in src_wrappers:
                wrapper_seed[bind_name] = src_wrappers[a.name]
    return alias_seed, wrapper_seed


def _propagate_range_aliases_across_modules(parsed_trees: dict, import_resolver) -> None:
    """Re-apply alias / wrapper rewrites with cross-module seeds."""
    while True:
        changed = False
        for module_name, (tree, _filename, preprocessor) in parsed_trees.items():
            alias_seed, wrapper_seed = _compute_range_seed(tree, import_resolver)
            before = _snapshot_exports(preprocessor)
            preprocessor.apply_range_rewrites(tree,
                                              alias_seed=alias_seed,
                                              wrapper_seed=wrapper_seed)
            after = _snapshot_exports(preprocessor)
            if after != before:
                import_resolver.module_exports[module_name] = after
                changed = True
        if not changed:
            return


def snapshot_exports(
    preprocessor: Preprocessor, ) -> tuple[set[str], dict[str, str], set[str] | None]:
    """Public façade for exporter snapshotting."""
    return _snapshot_exports(preprocessor)


def compute_range_seed(
    module_node: ast.Module,
    import_resolver,
) -> tuple[set[str], dict[str, str]]:
    """Public façade for range-seed projection."""
    return _compute_range_seed(module_node, import_resolver)


def propagate_range_aliases_across_modules(parsed_trees: dict, import_resolver) -> None:
    """Public façade for fixed-point alias propagation."""
    _propagate_range_aliases_across_modules(parsed_trees, import_resolver)
