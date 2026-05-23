"""Parser package facade for the Python frontend.

Keep this module lightweight: import heavy parser code lazily so importing
`parser` as a package does not execute parser-side setup eagerly.
"""

from __future__ import annotations

from typing import Any

__all__ = [
    "main",
    "parse_file",
    "parse_file_canonicalised",
    "generate_ast_json",
]


def main(*args: Any, **kwargs: Any):
    from .parser import main as _main
    return _main(*args, **kwargs)


def parse_file(*args: Any, **kwargs: Any):
    from .parser import parse_file as _parse_file
    return _parse_file(*args, **kwargs)


def parse_file_canonicalised(*args: Any, **kwargs: Any):
    from .parser import parse_file_canonicalised as _parse_file_canonicalised
    return _parse_file_canonicalised(*args, **kwargs)


def generate_ast_json(*args: Any, **kwargs: Any):
    from .parser import generate_ast_json as _generate_ast_json
    return _generate_ast_json(*args, **kwargs)
