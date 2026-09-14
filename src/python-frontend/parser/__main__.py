"""CLI entry point for the python-frontend parser package.

Kept parseable by Python 2: this module is what a mis-set interpreter reaches
first, and it can only report the version if it compiles there. Everything it
imports is Python 3 only, so the check has to come before the imports rather
than in bootstrap.ensure_python3 (issue #1967). Keeping it here also means
esbmc does not have to spawn a second interpreter to ask the same question.
"""
import sys

if sys.version_info[0] != 3:
    sys.stderr.write("ERROR: ESBMC's Python frontend requires Python 3 (this interpreter, "
                     "%s, reports version %d.%d).\nRe-run with --python <path-to-python3>.\n" %
                     (sys.executable, sys.version_info[0], sys.version_info[1]))
    sys.exit(1)

import importlib
import os


def _resolve_main():
    if __package__:
        return importlib.import_module(__package__ + ".parser").main

    # Support direct execution: ``python parser/__main__.py ...``
    parser_dir = os.path.dirname(os.path.abspath(__file__))
    python_frontend_dir = os.path.dirname(parser_dir)
    if python_frontend_dir not in sys.path:
        sys.path.insert(0, python_frontend_dir)
    return importlib.import_module("parser.parser").main


if __name__ == "__main__":
    _resolve_main()()
