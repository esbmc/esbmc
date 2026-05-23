"""CLI entry point for the python-frontend parser package."""
from __future__ import annotations

import importlib
import os
import sys


def _resolve_main():
    if __package__:
        return importlib.import_module(f"{__package__}.parser").main

    # Support direct execution: ``python parser/__main__.py ...``
    parser_dir = os.path.dirname(os.path.abspath(__file__))
    python_frontend_dir = os.path.dirname(parser_dir)
    if python_frontend_dir not in sys.path:
        sys.path.insert(0, python_frontend_dir)
    return importlib.import_module("parser.parser").main


if __name__ == "__main__":
    _resolve_main()()
