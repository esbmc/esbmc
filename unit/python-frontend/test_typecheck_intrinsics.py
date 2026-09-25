import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PARSER_DIR = os.path.join(ROOT, "src", "python-frontend", "parser")

if PARSER_DIR not in sys.path:
    sys.path.insert(0, PARSER_DIR)

# pylint: disable=wrong-import-position
from typecheck import _without_intrinsic_diagnostics


def test_intrinsic_only_report_is_dropped():
    report = ('main.py:2: error: Name "__ESBMC_requires" is not defined'
              '  [name-defined]\n'
              'main.py:3: error: Name "__ESBMC_return_value" is not defined'
              '  [name-defined]\n'
              'main.py:4: error: Name "nondet_int" is not defined'
              '  [name-defined]\n'
              'Found 3 errors in 1 file (checked 1 source file)\n')

    assert _without_intrinsic_diagnostics(1, report) == (0, "")


def test_misspelled_intrinsic_still_reaches_the_report():
    """The real gap: a name that looks like an intrinsic but is not one."""
    report = ('main.py:2: error: Name "__ESBMC_ensures" is not defined'
              '  [name-defined]\n'
              'main.py:3: error: Name "__ESBMC_assme" is not defined'
              '  [name-defined]\n'
              'main.py:4: error: Name "nondet_intt" is not defined'
              '  [name-defined]\n'
              'Found 3 errors in 1 file (checked 1 source file)\n')

    status, kept = _without_intrinsic_diagnostics(1, report)

    assert status == 1
    assert "__ESBMC_assme" in kept
    assert "nondet_intt" in kept
    assert "__ESBMC_ensures" not in kept


def test_mypy_failure_without_error_lines_is_preserved():
    report = "mypy: can't read file 'missing.py': No such file or directory\n"

    assert _without_intrinsic_diagnostics(2, report) == (2, report)


def test_non_name_errors_are_untouched():
    report = ('main.py:2: error: Incompatible return value type'
              '  [return-value]\n'
              'Found 1 error in 1 file (checked 1 source file)\n')

    status, kept = _without_intrinsic_diagnostics(1, report)

    assert status == 1
    assert "Incompatible return value type" in kept
