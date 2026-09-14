"""parser/__main__.py must stay parseable by Python 2.

esbmc no longer spawns a second interpreter to read sys.version_info: the entry
module reports the version itself. That only works if a Python 2 interpreter can
compile the module far enough to reach the check, which is what broke in #1967 --
an f-string made the whole module a SyntaxError before bootstrap.ensure_python3
could run.
"""
import ast
import pathlib

ENTRY = (pathlib.Path(__file__).resolve().parents[2] / "src" / "python-frontend" / "parser" /
         "__main__.py")


def _tree():
    return ast.parse(ENTRY.read_text(encoding="utf-8"))


def test_entry_module_has_no_python3_only_syntax():
    offenders = []
    for node in ast.walk(_tree()):
        if isinstance(node, ast.JoinedStr):
            offenders.append(f"f-string at line {node.lineno}")
        elif isinstance(node, ast.AnnAssign):
            offenders.append(f"variable annotation at line {node.lineno}")
        elif isinstance(node, ast.FunctionDef):
            if node.returns or any(a.annotation for a in node.args.args):
                offenders.append(f"annotated def {node.name}")
        elif isinstance(node, ast.ImportFrom) and node.module == "__future__":
            offenders.append(f"__future__ import at line {node.lineno}")
    assert not offenders, f"{ENTRY.name} is not Python 2 parseable: " + ", ".join(offenders)


def test_version_guard_precedes_every_other_import():
    body = _tree().body
    guards = [
        i for i, n in enumerate(body)
        if isinstance(n, ast.If) and "version_info" in ast.dump(n.test)
    ]
    assert guards, f"no sys.version_info guard in {ENTRY.name}"

    imports = [
        i for i, n in enumerate(body) if isinstance(n, (ast.Import, ast.ImportFrom))
        and not (isinstance(n, ast.Import) and any(a.name == "sys" for a in n.names))
    ]
    assert all(
        i > guards[0]
        for i in imports), ("an import runs before the version guard, so Python 2 would fail on it "
                            "instead of reporting the version")
