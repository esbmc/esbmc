"""Bootstrap helpers for parser package/script execution."""

from __future__ import annotations

import importlib
import os
import sys
from dataclasses import dataclass
from functools import lru_cache
from types import ModuleType
from typing import Final

__all__ = [
    "ParserModuleDeps",
    "ensure_python3",
    "ensure_python_frontend_on_path",
    "load_parser_module_deps",
]


@dataclass(frozen=True)
class ParserModuleDeps:
    """Resolved parser submodules used by parser façade orchestration."""
    constant_annotations: ModuleType
    json_emitter: ModuleType
    parser_cli: ModuleType
    range_alias_pipeline: ModuleType
    import_resolver: ModuleType
    rewrite_re: ModuleType
    threading_lowering: ModuleType

_PARSER_SUBMODULES: Final[tuple[str, ...]] = (
    "constant_annotations",
    "json_emitter",
    "parser_cli",
    "range_alias_pipeline",
    "import_resolver",
    "rewrite_re",
    "threading_lowering",
)


def ensure_python3() -> None:
    """Exit with a clear message when running under Python 2."""
    if sys.version_info[0] == 3:
        return
    # pylint: disable=consider-using-f-string
    print("Python version: {}.{}.{}".format(sys.version_info.major, sys.version_info.minor,
                                            sys.version_info.micro))
    print("ERROR: Please ensure Python 3 is available in your environment.")
    sys.exit(1)


def ensure_python_frontend_on_path(file_path: str) -> None:
    """Ensure python-frontend dir is importable for direct script execution."""
    parser_dir = os.path.dirname(os.path.abspath(file_path))
    python_frontend_dir = os.path.dirname(parser_dir)
    if python_frontend_dir not in sys.path:
        sys.path.insert(0, python_frontend_dir)


def _import_with_fallback(package: str | None, module_name: str) -> ModuleType:
    """Import ``module_name`` from package when present, else absolute fallback."""
    if package:
        return importlib.import_module(f"{package}.{module_name}")
    return importlib.import_module(module_name)


@lru_cache(maxsize=2)
def load_parser_module_deps(package: str | None) -> ParserModuleDeps:
    """Resolve parser submodules for package and script execution modes."""
    modules = {
        name: _import_with_fallback(package, name)
        for name in _PARSER_SUBMODULES
    }
    return ParserModuleDeps(
        constant_annotations=modules["constant_annotations"],
        json_emitter=modules["json_emitter"],
        parser_cli=modules["parser_cli"],
        range_alias_pipeline=modules["range_alias_pipeline"],
        import_resolver=modules["import_resolver"],
        rewrite_re=modules["rewrite_re"],
        threading_lowering=modules["threading_lowering"],
    )
