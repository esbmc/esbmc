"""Tests for the exception-site census (src/python-frontend/parser/exception_census.py)."""
import hashlib
import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
PY_FRONTEND_DIR = os.path.join(ROOT, "src", "python-frontend")

if PY_FRONTEND_DIR not in sys.path:
    sys.path.insert(0, PY_FRONTEND_DIR)
if os.path.dirname(__file__) not in sys.path:
    sys.path.insert(0, os.path.dirname(__file__))

# The local frontend `parser` package shadows the removed stdlib module.
# pylint: disable=wrong-import-position,deprecated-module,wrong-import-order
from parser import exception_census as census
from parser import exception_census_tables as tables
import exception_census_cases

FAMILY_TYPES = {
    "arithmetic": ("ZeroDivisionError", "FloatingPointError"),
    "data-lookup": ("KeyError", "IndexError"),
}


def write_task(directory, files):
    """Write the files of one case; main.py is the entry point."""
    for name, text in files.items():
        path = os.path.join(directory, name)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(text)
    return os.path.join(directory, "main.py")


def census_of(tmp_path, files, family):
    """The census of the entry file, looking only in its own directory."""
    main = write_task(str(tmp_path), files)
    result = census.run_one(main, family, "dir")
    return result


@pytest.mark.usefixtures("python_314")
@pytest.mark.parametrize("name,family,accepted,files", exception_census_cases.CASES)
def test_attack_and_control_programs(tmp_path, name, family, accepted, files):
    result = census_of(tmp_path, files, family)
    assert (not result.reports) == accepted, (name, result.reports[:3])


@pytest.fixture(name="python_314")
def fixture_python_314(monkeypatch):
    """The interpreter the tables were checked against, whatever runs the tests."""
    monkeypatch.setattr(census, "running_python", lambda: (3, 14))


@pytest.mark.usefixtures("python_314")
def test_a_program_with_no_site_gives_a_summary_line(tmp_path):
    main = write_task(str(tmp_path), {"main.py": "x = len('ab') / 2\n"})
    line = census.census_summary(main, ["ZeroDivisionError"])
    assert line.startswith("Exception census: no construct of ZeroDivisionError")


@pytest.mark.usefixtures("python_314")
@pytest.mark.parametrize(
    "types",
    [[], ["AssertionError"], ["TypeError", "AttributeError"], ["ZeroDivisionError", "KeyError"]])
def test_only_the_two_families_are_covered(tmp_path, types):
    main = write_task(str(tmp_path), {"main.py": "x = 1\n"})
    assert census.census_summary(main, types) is None


@pytest.mark.usefixtures("python_314")
def test_one_type_of_a_family_uses_the_whole_family(tmp_path):
    main = write_task(str(tmp_path), {"main.py": "d = {}\nx = d[1]\n"})
    assert census.census_summary(main, ["IndexError"]) is None


@pytest.mark.usefixtures("python_314")
def test_a_failing_census_declines_instead_of_raising(tmp_path, monkeypatch):
    main = write_task(str(tmp_path), {"main.py": "x = 1\n"})

    def boom(_self):
        raise RecursionError("deep")

    monkeypatch.setattr(census.Census, "run", boom)
    assert census.census_summary(main, ["ZeroDivisionError"]) is None


@pytest.mark.usefixtures("python_314")
def test_an_unparsable_program_declines(tmp_path):
    main = write_task(str(tmp_path), {"main.py": "def f(:\n"})
    assert census.census_summary(main, ["ZeroDivisionError"]) is None


@pytest.mark.usefixtures("python_314")
@pytest.mark.parametrize("version", [(3, 10), (3, 12), (3, 13), (3, 15)])
def test_an_interpreter_the_tables_were_not_checked_against_declines(tmp_path, monkeypatch,
                                                                     version):
    main = write_task(str(tmp_path), {"main.py": "x = 1\n"})
    monkeypatch.setattr(census, "running_python", lambda: version)
    assert census.census_summary(main, ["ZeroDivisionError"]) is None


def test_deeply_nested_input_declines(tmp_path):
    main = write_task(str(tmp_path), {"main.py": "x = " + "(" * 400 + "1" + ")" * 400 + "\n"})
    assert census.census_summary(main, ["ZeroDivisionError"]) is None


SV_TASK = "import _sv_verifier\nn = _sv_verifier.nondet_int()\n"


def test_sv_verifier_absent_is_the_audited_module(tmp_path):
    result = census_of(tmp_path, {"main.py": SV_TASK}, "arithmetic")
    assert not result.reports


def test_sv_verifier_next_to_the_program_must_match_the_audited_hash(tmp_path, monkeypatch):
    other = "def nondet_int():\n    return 0\n"
    files = {"main.py": SV_TASK, "_sv_verifier.py": other}
    assert census_of(tmp_path, files, "arithmetic").reports
    digest = hashlib.sha256(other.encode()).hexdigest()
    monkeypatch.setattr(tables, "AUDITED_SV_VERIFIER", frozenset({digest}))
    assert not census_of(tmp_path, files, "arithmetic").reports


def test_any_star_import_declines(tmp_path):
    source = "from _sv_verifier import *\nn = nondet_int()\n"
    result = census_of(tmp_path, {"main.py": source}, "arithmetic")
    assert result.reports[0][3] == "star-import"


def test_roots_follow_pythonpath(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "helper.py").write_text("x = 1 / len(())\n", encoding="utf-8")
    task = tmp_path / "task"
    task.mkdir()
    main = write_task(str(task), {"main.py": "import helper\n"})
    monkeypatch.setenv("PYTHONPATH", str(lib))
    assert census.run_one(main, "arithmetic").reports[0][3] == "divisor:Div"
    monkeypatch.delenv("PYTHONPATH")
    assert census.run_one(main, "arithmetic").reports[0][3] == "import:helper"


def test_a_symlinked_program_looks_beside_its_target(tmp_path):
    real = tmp_path / "real"
    real.mkdir()
    (real / "helper.py").write_text("x = 1 / len(())\n", encoding="utf-8")
    (real / "main.py").write_text("import helper\n", encoding="utf-8")
    link = tmp_path / "link"
    link.mkdir()
    os.symlink(real / "main.py", link / "main.py")
    result = census.run_one(str(link / "main.py"), "arithmetic", "dir")
    assert result.reports[0][3] == "divisor:Div"


@pytest.mark.parametrize("compiled", [
    "helper.pyc", "helper.cpython-314-x86_64-linux-gnu.so", "helper.pyd",
    "__pycache__/helper.cpython-314.pyc"
])
def test_compiled_code_next_to_an_imported_module_declines(tmp_path, compiled):
    files = {"main.py": "import helper\n", "helper.py": "x = 1\n", compiled: ""}
    result = census_of(tmp_path, files, "arithmetic")
    assert any(r[3].startswith("compiled-module") for r in result.reports)


def test_a_compiled_package_initialiser_declines(tmp_path):
    files = {
        "main.py": "import pkg\n",
        "pkg/__init__.py": "",
        "pkg/__pycache__/__init__.cpython-314.pyc": ""
    }
    assert census_of(tmp_path, files, "arithmetic").reports


@pytest.mark.parametrize(
    "compiled",
    ["_sv_verifier.pyc", "_sv_verifier.cpython-314.so", "__pycache__/_sv_verifier.cpython-314.pyc"])
def test_compiled_sv_verifier_declines(tmp_path, compiled):
    files = {"main.py": SV_TASK, compiled: ""}
    assert census_of(tmp_path, files, "arithmetic").reports


def test_compiled_code_for_a_stdlib_name_declines(tmp_path):
    files = {"main.py": "import math\n", "math.cpython-314.so": ""}
    assert census_of(tmp_path, files, "arithmetic").reports


def test_source_next_to_a_program_without_compiled_code_is_fine(tmp_path):
    files = {"main.py": "import helper\n", "helper.py": "x = 1\n", "other.pyc": ""}
    assert not census_of(tmp_path, files, "arithmetic").reports


def test_an_empty_pythonpath_entry_is_the_working_directory(tmp_path, monkeypatch):
    lib = tmp_path / "lib"
    lib.mkdir()
    (lib / "helper.py").write_text("x = 1 / len(())\n", encoding="utf-8")
    task = tmp_path / "task"
    task.mkdir()
    main = write_task(str(task), {"main.py": "import helper\n"})
    monkeypatch.chdir(lib)
    monkeypatch.setenv("PYTHONPATH", os.pathsep + "/nonexistent")
    assert census.run_one(main, "arithmetic").reports[0][3] == "divisor:Div"


def test_an_archive_on_the_search_path_declines(tmp_path, monkeypatch):
    archive = tmp_path / "libs.zip"
    archive.write_bytes(b"PK")
    task = tmp_path / "task"
    task.mkdir()
    main = write_task(str(task), {"main.py": "x = 1\n"})
    monkeypatch.setenv("PYTHONPATH", str(archive))
    assert census.run_one(main, "arithmetic").reports[0][3] == "path-entry-not-a-directory"


def test_a_missing_search_path_entry_is_ignored(tmp_path, monkeypatch):
    main = write_task(str(tmp_path), {"main.py": "x = 1\n"})
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "nowhere"))
    assert not census.run_one(main, "arithmetic").reports


RULE_CASES = [
    ("library-class-value", {
        "main.py": "import random\nR = random.Random\n"
    }),
    ("nested-import", {
        "main.py": "def g():\n    import helper\n    return helper.x\n",
        "helper.py": "x = 1\n"
    }),
    ("nested-import", {
        "main.py": "class C:\n    import helper\n\n\nx = C.helper.x\n",
        "helper.py": "x = 1\n"
    }),
    ("nested-import", {
        "main.py": "def g():\n    from pkg import helper\n    return helper\n",
        "pkg/__init__.py": "",
        "pkg/helper.py": "x = 1\n"
    }),
    ("class-base", {
        "main.py": "from typing import List as ABC\n\n\nclass D(ABC):\n    pass\n"
    }),
    ("class-base", {
        "main.py": "from collections import deque as Protocol\n\n\nclass D(Protocol):\n    pass\n"
    }),
    ("class-base", {
        "main.py":
        "import typing\n\nclass N:\n    Generic = dict\n\n\nclass D(N.Generic):\n    pass\n"
    }),
    ("class-base", {
        "main.py": "import abc\nabc = dict\n\n\nclass D(abc.ABC):\n    pass\n"
    }),
    ("import", {
        "main.py": "import enum\n"
    }),
    ("attr", {
        "main.py": "import functools\nfunctools.wraps\n"
    }),
    ("store-universe-attr", {
        "main.py": "import random\nr = random.Random()\nr.choice = None\n"
    }),
    ("self-import", {
        "main.py": "import main as me\n"
    }),
    ("star-import", {
        "main.py": "from helper import *\n",
        "helper.py": "x = 1\n"
    }),
    ("init-binds-submodule", {
        "main.py": "import pkg.sub\n",
        "pkg/__init__.py": "from . import sub\n",
        "pkg/sub.py": "x = 1\n"
    }),
    ("module-attribute-store", {
        "main.py": "import helper\nhelper.x = 1\n",
        "helper.py": "x = 0\n"
    }),
    ("module-attribute-store", {
        "main.py": "import helper\ndel helper.x\n",
        "helper.py": "x = 0\n"
    }),
    ("module-value", {
        "main.py": "import helper\nm = helper\n",
        "helper.py": "x = 0\n"
    }),
    ("module-value", {
        "main.py": "import pkg.sub\nm = pkg.sub\n",
        "pkg/__init__.py": "",
        "pkg/sub.py": "x = 0\n"
    }),
    ("module-value", {
        "main.py": "import mod1\nx = mod1.h\n",
        "mod1.py": "import helper as h\n",
        "helper.py": "x = 0\n"
    }),
    ("module-value", {
        "main.py": "import helper\nprint(helper)\n",
        "helper.py": "x = 0\n"
    }),
]


@pytest.mark.parametrize("key,files", RULE_CASES)
def test_each_module_rule_reports_its_own_key(tmp_path, key, files):
    reasons = {r[3].split(":")[0] for r in census_of(tmp_path, files, "data-lookup").reports}
    assert key in reasons


def test_a_module_held_in_a_variable_is_the_only_report_for_a_store_through_it(tmp_path):
    files = {
        "helper.py": "def go():\n    return 1\n",
        "mod1.py": "import helper as h\n\ndef go():\n    return h.go()\n",
        "main.py": "import random\nimport mod1\n\nm = mod1\nm.h = random.Random()\nmod1.go()\n",
    }
    reasons = {r[3] for r in census_of(tmp_path, files, "data-lookup").reports}
    assert reasons == {"module-value"}


def test_plain_attribute_stores_and_module_reads_stay_allowed(tmp_path):
    files = {
        "main.py": "import helper\n\nclass P:\n    def __init__(self):\n        self.x = 1\n"
        "\n\np = P()\np.y = helper.z\n",
        "helper.py": "z = 2\n"
    }
    assert not census_of(tmp_path, files, "data-lookup").reports


LIBRARY_BASES = [
    "from abc import ABC\n\n\nclass A(ABC):\n    pass\n",
    "import abc\n\n\nclass A(abc.ABC):\n    pass\n",
    "import abc as a\n\n\nclass A(a.ABC):\n    pass\n",
    "from typing import Generic, TypeVar\nT = TypeVar('T')\n\n\nclass G(Generic[T]):\n    pass\n",
    "import typing\nT = typing.TypeVar('T')\n\n\nclass G(typing.Generic[T]):\n    pass\n",
    "class E(Exception):\n    pass\n\n\nclass F(E):\n    pass\n",
    "class O(object):\n    pass\n",
]


@pytest.mark.parametrize("source", LIBRARY_BASES)
def test_exact_library_bases_stay_allowed(tmp_path, source):
    assert not census_of(tmp_path, {"main.py": source}, "data-lookup").reports
