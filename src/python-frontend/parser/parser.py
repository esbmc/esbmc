# pylint: disable=wrong-import-position
"""Thin façade/orchestrator for the Python frontend parser package."""

from __future__ import annotations

import ast
import importlib

__all__ = [
    "emit_file_as_json",
    "emit_module_json",
    "generate_ast_json",
    "main",
    "parse_file",
    "parse_file_canonicalised",
]

if __package__:
    bootstrap = importlib.import_module(f"{__package__}.bootstrap")
    typecheck = importlib.import_module(f"{__package__}.typecheck")
else:
    bootstrap = importlib.import_module("bootstrap")
    typecheck = importlib.import_module("typecheck")

ensure_python3 = bootstrap.ensure_python3
ensure_python_frontend_on_path = bootstrap.ensure_python_frontend_on_path
load_parser_module_deps = bootstrap.load_parser_module_deps
run_mypy_strict = typecheck.run_mypy_strict

ensure_python3()
ensure_python_frontend_on_path(__file__)

from preprocessor import Preprocessor

_DEPS = load_parser_module_deps(__package__)

add_type_annotation = _DEPS.constant_annotations.add_type_annotation
annotate_constant_node = _DEPS.constant_annotations.annotate_constant_node
tag_bignum_constants = _DEPS.constant_annotations.tag_bignum_constants
_emit_file_as_json = _DEPS.json_emitter.emit_file_as_json
_emit_module_json = _DEPS.json_emitter.emit_module_json
_generate_ast_json = _DEPS.json_emitter.generate_ast_json
EmitPipelineDeps = _DEPS.json_emitter.EmitPipelineDeps
JsonEmitterDeps = _DEPS.json_emitter.JsonEmitterDeps
cli_main = _DEPS.parser_cli.main
CliDeps = _DEPS.parser_cli.CliDeps
compute_range_seed = _DEPS.range_alias_pipeline.compute_range_seed
propagate_range_aliases = _DEPS.range_alias_pipeline.propagate_range_aliases_across_modules
snapshot_exports = _DEPS.range_alias_pipeline.snapshot_exports
parse_file_canonicalised = _DEPS.range_alias_pipeline.parse_file_canonicalised
import_resolver = _DEPS.import_resolver
rewrite_re_match_attribute_calls = _DEPS.rewrite_re.rewrite_re_match_attribute_calls
threading_lowering = _DEPS.threading_lowering


def parse_file(filename: str) -> tuple[ast.AST, Preprocessor]:
    """Open, parse, and run Preprocessor on a Python source file."""
    with open(filename, "r", encoding="utf-8") as src:
        tree = ast.parse(src.read())
    preprocessor = Preprocessor(filename)
    tree = preprocessor.visit(tree)
    return tree, preprocessor


def generate_ast_json(
    tree: ast.AST,
    python_filename: str,
    elements_to_import,
    output_dir: str,
    module_qualname: str | None = None,
) -> None:
    """Wrapper around json emitter with parser-scoped dependencies."""
    return _generate_ast_json(
        tree,
        python_filename,
        elements_to_import,
        output_dir,
        module_qualname=module_qualname,
        deps=JsonEmitterDeps(
            import_resolver=import_resolver,
            tag_bignum_constants=tag_bignum_constants,
        ),
    )


def emit_file_as_json(
    filename: str,
    output_dir: str,
    module_qualname: str | None = None,
    elements_to_import=None,
) -> None:
    """Wrapper around file JSON emission with parser-scoped dependencies."""
    return _emit_file_as_json(
        filename,
        output_dir,
        module_qualname,
        elements_to_import,
        deps=EmitPipelineDeps(
            parse_file=parse_file,
            generate_ast_json_fn=generate_ast_json,
        ),
    )


def emit_module_json(
    module_qualname: str,
    output_dir: str,
    elements_to_import=None,
) -> None:
    """Wrapper around module JSON emission with parser-scoped dependencies."""
    return _emit_module_json(
        module_qualname,
        output_dir,
        elements_to_import,
        deps=EmitPipelineDeps(
            import_resolver=import_resolver,
            emit_file_as_json_fn=emit_file_as_json,
        ),
    )


def main() -> int | None:
    """Run parser CLI orchestration with assembled dependencies."""
    return cli_main(deps=CliDeps(
        run_mypy_strict=run_mypy_strict,
        parse_file_canonicalised=parse_file_canonicalised,
        generate_ast_json_fn=generate_ast_json,
        annotate_constant_node=annotate_constant_node,
        add_type_annotation=add_type_annotation,
        import_resolver=import_resolver,
        rewrite_re_match_attribute_calls=rewrite_re_match_attribute_calls,
        threading_lowering=threading_lowering,
        preprocessor_cls=Preprocessor,
        snapshot_exports=snapshot_exports,
        compute_range_seed=compute_range_seed,
        propagate_range_aliases_across_modules=propagate_range_aliases,
    ))


if __name__ == "__main__":
    main()
