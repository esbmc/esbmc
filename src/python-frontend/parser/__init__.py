"""Parser package facade for the Python frontend.

Keep this module lightweight: import heavy parser code lazily so importing
`parser` as a package does not execute parser-side setup eagerly.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = [
    "main",
    "parse_file",
    "parse_file_canonicalised",
    "generate_ast_json",
]


def _impl():
    return importlib.import_module(f"{__name__}.parser")


def main(*args: Any, **kwargs: Any):
    return _impl().main(*args, **kwargs)


def parse_file(*args: Any, **kwargs: Any):
    return _impl().parse_file(*args, **kwargs)


def parse_file_canonicalised(*args: Any, **kwargs: Any):
    return _impl().parse_file_canonicalised(*args, **kwargs)


def generate_ast_json(*args: Any, **kwargs: Any):
    return _impl().generate_ast_json(*args, **kwargs)
