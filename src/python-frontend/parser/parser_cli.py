"""CLI orchestration for the Python parser frontend."""

from __future__ import annotations

import ast
import glob
import os
import shutil
import sys
from dataclasses import dataclass
from typing import Any, Callable, Protocol, TypeAlias

__all__ = ["CliDeps", "check_dependencies", "check_usage", "main", "select_threading_model"]

RunMypyStrictFn: TypeAlias = Callable[[str], tuple[int, str]]
ParseFileCanonicalisedFn: TypeAlias = Callable[[str], tuple[ast.AST, object]]
GenerateAstJsonFn: TypeAlias = Callable[[ast.AST, str, object, str], None]
AnnotateConstantFn: TypeAlias = Callable[[ast.AST], None]
AddTypeAnnotationFn: TypeAlias = Callable[[ast.Assign], None]
SnapshotExportsFn: TypeAlias = Callable[[object], tuple[set[str], dict[str, str], set[str] | None]]
ComputeRangeSeedFn: TypeAlias = Callable[[ast.Module, object], tuple[set[str], dict[str, str]]]
PropagateRangeAliasesFn: TypeAlias = Callable[[dict, object], None]
RewriteReFn: TypeAlias = Callable[[ast.Module], None]


class ImportResolverLike(Protocol):
    """Structural type for resolver services used by parser CLI."""
    # pylint: disable=missing-function-docstring

    module_exports: dict[str, tuple[set[str], dict[str, str], set[str] | None]]
    rewrite_relative_import: Callable[..., Any]
    ResolverCallbacks: type

    def reset_state(self) -> None:
        ...

    def process_imports(self, node: ast.Import | ast.ImportFrom, output_dir: str) -> None:
        ...

    def process_collected_imports(self, output_dir: str, callbacks: object) -> None:
        ...

    def detect_and_process_submodules(
        self,
        node: ast.AST,
        processed_submodules: set[str],
        output_dir: str,
        generate_ast_json_fn: GenerateAstJsonFn,
    ) -> None:
        ...

    def is_imported_model(self, module_name: str) -> bool:
        ...


class ThreadingLoweringLike(Protocol):
    """Structural type for threading validation/lowering entry points."""
    # pylint: disable=missing-function-docstring

    def reject_unsupported_threading_usage(self, tree: ast.AST, source_filename: str) -> None:
        ...

    def validate_threading_thread_usage(self, tree: ast.Module, source_filename: str) -> None:
        ...

    def lower_threading_thread_usage(self, tree: ast.Module, source_filename: str) -> None:
        ...


@dataclass(frozen=True)
# pylint: disable=too-many-instance-attributes
class CliDeps:
    """Dependency bundle injected by parser.py orchestration."""
    run_mypy_strict: RunMypyStrictFn
    parse_file_canonicalised: ParseFileCanonicalisedFn
    generate_ast_json_fn: GenerateAstJsonFn
    annotate_constant_node: AnnotateConstantFn
    add_type_annotation: AddTypeAnnotationFn
    import_resolver: ImportResolverLike
    rewrite_re_match_attribute_calls: RewriteReFn
    threading_lowering: ThreadingLoweringLike
    preprocessor_cls: Callable[[str], Any]
    snapshot_exports: SnapshotExportsFn
    compute_range_seed: ComputeRangeSeedFn
    propagate_range_aliases_across_modules: PropagateRangeAliasesFn


@dataclass(frozen=True)
class _ResolverCallbacksDeps:
    """Typed bundle used to assemble ResolverCallbacks wiring."""
    parse_file_canonicalised: ParseFileCanonicalisedFn
    rewrite_relative_import: Callable[..., Any]
    snapshot_exports: SnapshotExportsFn
    propagate_range_aliases: Callable[..., Any]
    generate_ast_json: GenerateAstJsonFn


def _read_ast_from_file(filename: str) -> ast.Module:
    """Read and parse a Python source file into an AST module."""
    with open(filename, "r", encoding="utf-8") as source:
        return ast.parse(source.read())


def check_usage() -> None:
    """Validate CLI args and fail with usage message when invalid."""
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print("Usage: python parser/__main__.py <file path> <output directory> "
              "[--deadlock-check]")
        sys.exit(2)
    if len(sys.argv) == 4 and sys.argv[3] != "--deadlock-check":
        print(f"Unknown flag: {sys.argv[3]}")
        sys.exit(2)


def check_dependencies() -> None:
    """Warn when optional tools for richer diagnostics are missing."""
    if shutil.which("mypy") is None:
        print("Warning: mypy not found on PATH; type checking will be skipped.")
        print("  Install with: pip install mypy  or  pipx install mypy")


def select_threading_model(output_dir: str, deadlock_check: bool) -> None:
    """Activate deadlock threading model variant when requested."""
    if not deadlock_check:
        return
    models_dir = os.path.join(output_dir, "models")
    src = os.path.join(models_dir, "threading_deadlock.py")
    dst = os.path.join(models_dir, "threading.py")
    if os.path.exists(src):
        shutil.copyfile(src, dst)


def _emit_model_jsons(
    output_dir: str,
    *,
    import_resolver: ImportResolverLike,
    generate_ast_json_fn: GenerateAstJsonFn,
) -> None:
    """Generate AST JSON for each memory-model module in ``output_dir/models``."""
    models_dir = os.path.join(output_dir, "models")
    for python_file in glob.glob(os.path.join(models_dir, "*.py")):
        filename = os.path.basename(python_file)
        module_name = filename[:-3]

        if module_name == "threading_deadlock":
            continue

        if import_resolver.is_imported_model(module_name) and module_name != "typing":
            continue

        with open(python_file, encoding="utf-8") as model:
            model_tree = ast.parse(model.read())
            generate_ast_json_fn(model_tree, filename, None, output_dir)


def main(*, deps: CliDeps) -> int | None:
    """Run parser CLI orchestration with explicit dependency injection."""
    check_usage()
    check_dependencies()

    import_resolver = deps.import_resolver
    import_resolver.reset_state()

    filename = sys.argv[1]
    output_dir = sys.argv[2]
    deadlock_check = len(sys.argv) == 4 and sys.argv[3] == "--deadlock-check"

    returncode, mypy_output = deps.run_mypy_strict(filename)
    if returncode != 0:
        print("\033[93m\nType checking warning:\033[0m")
        print(mypy_output)

    script_dir = os.path.dirname(os.path.abspath(filename))
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    os.makedirs(output_dir, exist_ok=True)
    select_threading_model(output_dir, deadlock_check)

    tree = _read_ast_from_file(filename)

    preprocessor = deps.preprocessor_cls(filename)
    preprocessor.prepare_module(tree)
    import_resolver.module_exports["__main__"] = deps.snapshot_exports(preprocessor)

    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_resolver.process_imports(node, output_dir)

    callbacks_deps = _ResolverCallbacksDeps(
        parse_file_canonicalised=deps.parse_file_canonicalised,
        rewrite_relative_import=import_resolver.rewrite_relative_import,
        snapshot_exports=deps.snapshot_exports,
        propagate_range_aliases=lambda parsed: deps.propagate_range_aliases_across_modules(
            parsed, import_resolver),
        generate_ast_json=deps.generate_ast_json_fn,
    )
    import_resolver.process_collected_imports(
        output_dir,
        import_resolver.ResolverCallbacks(
            parse_file_canonicalised=callbacks_deps.parse_file_canonicalised,
            rewrite_relative_import=callbacks_deps.rewrite_relative_import,
            snapshot_exports=callbacks_deps.snapshot_exports,
            propagate_range_aliases=callbacks_deps.propagate_range_aliases,
            generate_ast_json=callbacks_deps.generate_ast_json,
        ),
    )

    alias_seed, wrapper_seed = deps.compute_range_seed(tree, import_resolver)
    preprocessor.apply_range_rewrites(tree, alias_seed=alias_seed, wrapper_seed=wrapper_seed)
    tree = preprocessor.finalize_module(tree)

    processed_submodules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            deps.add_type_annotation(node)
        elif isinstance(node, ast.Constant):
            deps.annotate_constant_node(node)
        elif isinstance(node, ast.Attribute):
            import_resolver.detect_and_process_submodules(node, processed_submodules, output_dir,
                                                          deps.generate_ast_json_fn)

    deps.threading_lowering.reject_unsupported_threading_usage(tree, filename)
    deps.threading_lowering.validate_threading_thread_usage(tree, filename)
    deps.threading_lowering.lower_threading_thread_usage(tree, filename)

    deps.rewrite_re_match_attribute_calls(tree)

    deps.generate_ast_json_fn(tree, filename, None, output_dir)
    _emit_model_jsons(output_dir,
                      import_resolver=import_resolver,
                      generate_ast_json_fn=deps.generate_ast_json_fn)
