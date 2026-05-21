# pylint: disable=wrong-import-position
# Imports below the PY3 check are intentional: the check is a hard fail
# under Python 2 and must run before the Python-3-only imports (ast,
# importlib.util, etc.) to produce a clean error message instead of an
# ImportError stack trace.
#
# pylint: disable=c-extension-no-member
# `mypy.api.run` is a real (typed) entry point; pylint only flags it
# because mypy ships as a compiled extension and is not always installed
# in the lint environment.
from __future__ import annotations

import sys

# Detect the Python version
PY3 = sys.version_info[0] == 3

if not PY3:
    # pylint: disable=consider-using-f-string  # f-strings are a SyntaxError on Python 2
    print("Python version: {}.{}.{}".format(sys.version_info.major, sys.version_info.minor,
                                            sys.version_info.micro))
    print("ERROR: Please ensure Python 3 is available in your environment.")
    sys.exit(1)

import ast
import copy
import importlib.util
import json
import os
import glob
import base64
import shutil
import subprocess
import tempfile
from libs.ast2json import ast2json as ast2json_func
from preprocessor import Preprocessor

# Python ints are arbitrary precision; the JSON wire format used by the C++
# frontend stores them as numbers, which nlohmann::json silently truncates to
# double once they exceed uint64. Tag any int Constant whose value does not
# fit int64 so the C++ side can detect and diagnose the overflow without
# losing precision, and replace the numeric value with `null` — direct C++
# readers of Constant["value"] then bail via their `is_number_integer()`
# guard. The dedicated `get_literal()` path, plus the consteval and funcall
# pre-scan paths, check `_bigint` first and refuse to fold. Issue #4642
# tracks the bignum lowering that will eventually consume the tag.
#
# When a Constant sits in the `operand` position of UnaryOp(USub, …) Python's
# ast.parse splits a negative literal like ``-(2**63)`` into the surrounding
# minus and a positive Constant; the inner Constant is technically out of
# int64 range even though the final value is INT64_MIN. Take that one-level
# context into account so valid INT64_MIN literals are accepted while true
# out-of-range literals (signed or not) still trap.
_INT64_MIN = -(1 << 63)
_INT64_MAX = (1 << 63) - 1


def _fits_int64(v):
    return _INT64_MIN <= v <= _INT64_MAX


def _is_usub_unaryop(node):
    return (isinstance(node, dict)
            and node.get("_type") == "UnaryOp"
            and isinstance(node.get("op"), dict)
            and node["op"].get("_type") == "USub")


def _tag_bignum_constants(node, in_usub_operand=False):
    if isinstance(node, dict):
        if node.get("_type") == "Constant":
            v = node.get("value")
            if isinstance(v, int) and not isinstance(v, bool):
                effective = -v if in_usub_operand else v
                if not _fits_int64(effective):
                    node["_bigint"] = str(v)
                    node["value"] = None
            return  # Constants are leaves — no nested Constants inside
        is_usub = _is_usub_unaryop(node)
        for k, v in node.items():
            _tag_bignum_constants(
                v, in_usub_operand=(is_usub and k == "operand"))
    elif isinstance(node, list):
        for v in node:
            _tag_bignum_constants(v)


def run_mypy_strict(filename):
    """Run mypy as a subprocess when available; skip otherwise."""
    mypy_path = shutil.which("mypy")
    if mypy_path is None:
        return 0, ""

    with tempfile.TemporaryDirectory(prefix="esbmc-mypy-cache-") as cache_dir:
        result = subprocess.run(
            [mypy_path, "--strict", "--cache-dir", cache_dir, filename],
            capture_output=True,
            text=True,
            check=False,
        )
    output = result.stdout
    if result.stderr:
        output += result.stderr
    return result.returncode, output


def check_usage():
    if len(sys.argv) < 3 or len(sys.argv) > 4:
        print(
            "Usage: python astgen.py <file path> <output directory> "
            "[--deadlock-check]"
        )
        sys.exit(2)
    if len(sys.argv) == 4 and sys.argv[3] != "--deadlock-check":
        print(f"Unknown flag: {sys.argv[3]}")
        sys.exit(2)


def is_imported_model(module_name):
    models = [
        "math",
        "cmath",
        "os",
        "numpy",
        "esbmc",
        "decimal",
        "collections",
        "dataclasses",
        "typing",
        "time",
        "threading",
    ]
    return module_name in models


def is_unsupported_module(module_name):
    unsuported_modules = ["blah"]
    return module_name in unsuported_modules


# Names from the ``threading`` module that the generic threading reject
# pass tolerates. ``Lock`` is fully modelled today; ``Thread`` is handled
# by :func:`validate_threading_thread_usage` (which enforces the MVP
# structural constraints and refuses anything otherwise valid until the
# converter-side lowering lands — tracked in #4568). Anything else
# (RLock, Semaphore, Condition, Event, Barrier, Timer, ...) is rejected
# at parse time by :func:`reject_unsupported_threading_usage` so we
# never emit a half-modelled concurrency construct that could yield a
# silently wrong verification verdict.
SUPPORTED_THREADING_SYMBOLS = frozenset({"Lock", "Thread"})

# Thread keyword arguments outside MVP scope. Only ``target=`` and
# ``args=`` are accepted; the rest are refused at parse time so the
# converter cannot silently drop semantics that would change the
# verification verdict (e.g. dropping ``daemon=`` would change shutdown
# behaviour; dropping ``kwargs=`` would lose actual data the thread
# reads).
UNSUPPORTED_THREAD_KWARGS = frozenset({"daemon", "name", "kwargs", "group"})

# Methods on ``threading.Thread`` whose semantics the MVP lowering owns
# (calls are rewritten to pthread intrinsics). A user subclass that
# redefines ``start`` or ``join`` would have its override silently
# bypassed at spawn/join — refuse at parse time.
THREAD_OVERRIDE_REJECTED_METHODS = frozenset({"start", "join"})


def reject_unsupported_threading_usage(tree: ast.AST, source_filename: str) -> None:
    """Refuse to compile programs using unsupported ``threading`` names.

    Walks the AST for usages of names from the ``threading`` module that
    ESBMC does not yet model and exits with a clear error rather than
    silently emitting a weaker abstraction. The supported set is
    ``SUPPORTED_THREADING_SYMBOLS``. Detects three import shapes:

      ``import threading``         → ``threading.<X>`` attribute access
      ``import threading as t``    → ``t.<X>`` attribute access
      ``from threading import X``  → bare ``X`` reference

    ``from threading import *`` is refused outright because static name
    resolution would require importing the real ``threading`` module.
    """
    def fail(line, message: str) -> None:
        print(f"ERROR: {source_filename}:{line}: {message}")
        sys.exit(4)

    unsupported_message = (
        "is not yet supported by ESBMC. Only threading.Lock is "
        "currently modelled; Thread/RLock/Semaphore/Condition/Event/"
        "Barrier/Timer are tracked as follow-ups to the initial "
        "threading support."
    )

    module_aliases: set[str] = set()
    name_aliases: dict[str, str] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "threading":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "threading":
            for alias in node.names:
                if alias.name == "*":
                    fail(
                        node.lineno,
                        "'from threading import *' is not supported; "
                        "import names explicitly so ESBMC can verify "
                        "each one is modelled.",
                    )
                name_aliases[alias.asname or alias.name] = alias.name

    for node in ast.walk(tree):
        offending: str | None = None
        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id in module_aliases
            and node.attr not in SUPPORTED_THREADING_SYMBOLS
        ):
            offending = node.attr
        elif isinstance(node, ast.Name) and node.id in name_aliases:
            original = name_aliases[node.id]
            if original not in SUPPORTED_THREADING_SYMBOLS:
                offending = original

        if offending is not None:
            fail(
                getattr(node, "lineno", "?"),
                f"threading.{offending} {unsupported_message}",
            )


def _collect_thread_aliases(tree: ast.AST) -> tuple[set[str], set[str]]:
    """Return ``(module_aliases, thread_aliases)`` for the ``threading`` module.

    ``module_aliases`` are the names bound by ``import threading [as X]``;
    ``thread_aliases`` are the names bound by
    ``from threading import Thread [as X]``. The two sets disambiguate
    qualified (``X.Thread``) from bare (``X``) references at every Thread
    construction site.
    """
    module_aliases: set[str] = set()
    thread_aliases: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "threading":
                    module_aliases.add(alias.asname or alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module == "threading":
            for alias in node.names:
                if alias.name == "Thread":
                    thread_aliases.add(alias.asname or alias.name)
    return module_aliases, thread_aliases


def _is_thread_constructor(
    call_node: ast.Call,
    module_aliases: set[str],
    thread_aliases: set[str],
) -> bool:
    """Return ``True`` iff ``call_node`` constructs ``threading.Thread``.

    Recognises both ``threading.Thread(...)`` (qualified through an
    ``import threading [as X]`` alias) and ``Thread(...)`` (bare, through
    a ``from threading import Thread [as X]`` alias).
    """
    func = call_node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id in module_aliases and func.attr == "Thread"
    if isinstance(func, ast.Name):
        return func.id in thread_aliases
    return False


def _is_thread_construction_with_target_kw(
    value: ast.expr | None,
    module_aliases: set[str],
    thread_aliases: set[str],
) -> bool:
    """Return ``True`` iff ``value`` is a real ``Thread(target=...)`` call.

    Used by the threading-rewrite path to recognise the construction
    arm: a ``threading.Thread`` call with no positional arguments and a
    ``target=`` keyword. A user subclass whose ``__init__`` accepts a
    parameter named ``target`` is excluded by the ``_is_thread_constructor``
    check.
    """
    if not (isinstance(value, ast.Call) and not value.args):
        return False
    if not _is_thread_constructor(value, module_aliases, thread_aliases):
        return False
    return any(kw.arg == "target" for kw in value.keywords)


def _base_is_thread(
    base: ast.expr,
    module_aliases: set[str],
    thread_aliases: set[str],
) -> bool:
    """Return ``True`` iff ``base`` resolves to ``threading.Thread``.

    Recognises both ``threading.Thread`` (qualified through an
    ``import threading [as X]`` alias) and the bare name when it
    came in via ``from threading import Thread [as X]``.
    """
    if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
        return base.value.id in module_aliases and base.attr == "Thread"
    if isinstance(base, ast.Name):
        return base.id in thread_aliases
    return False


def _collect_thread_subclass_defs(
    tree: ast.AST,
    module_aliases: set[str],
    thread_aliases: set[str],
) -> dict[str, ast.ClassDef]:
    """Return ``{class_name: ClassDef}`` for module-top direct subclasses.

    Only direct subclasses (``class X(threading.Thread)``) at module
    scope are picked up. Transitive subclasses (``class B(A)`` where
    ``A`` is itself a Thread subclass) and nested subclass defs are out
    of MVP scope; the validator's C1 rule rejects nested defs and
    transitive subclassing fails the constructor-recognition check at
    the call site.
    """
    if not isinstance(tree, ast.Module):
        return {}
    out: dict[str, ast.ClassDef] = {}
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue
        if any(
            _base_is_thread(b, module_aliases, thread_aliases) for b in node.bases
        ):
            out[node.name] = node
    return out


def _is_subclass_constructor(
    call_node: ast.Call,
    subclass_names: set[str],
) -> bool:
    """Return ``True`` iff ``call_node`` constructs a tracked Thread subclass."""
    func = call_node.func
    return isinstance(func, ast.Name) and func.id in subclass_names


def _extract_name_binding(
    stmt: ast.stmt,
) -> tuple[str | None, ast.expr | None]:
    """Return ``(target_name, value)`` for ``Name = value`` / ``Name: T = value``.

    Returns ``(None, None)`` for tuple-targets, attribute-targets, augmented
    assigns, and any other statement shape. Used wherever we need to ask
    "is this a simple single-name binding, and what does it assign?"
    """
    if (
        isinstance(stmt, ast.Assign)
        and len(stmt.targets) == 1
        and isinstance(stmt.targets[0], ast.Name)
    ):
        return stmt.targets[0].id, stmt.value
    if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name):
        return stmt.target.id, stmt.value
    return None, None


def _call_class_name(call_node: ast.Call) -> str:
    """Return the class name from a ``Name(...)`` call for error messages."""
    func = call_node.func
    if isinstance(func, ast.Name):
        return func.id
    return "<expr>"


def _is_super_call_expr(node: ast.AST) -> bool:
    """Return ``True`` iff ``node`` is a call whose function is ``super().X``.

    Matches the AST shape ``Call(func=Attribute(value=Call(func=Name('super'))))``
    for any attribute. Used to enumerate every super-method call inside
    a subclass body so the validator can refuse the ones we cannot
    soundly strip.
    """
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    inner = func.value
    return (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == "super"
    )


def _is_super_init_call_stmt(stmt: ast.stmt) -> bool:
    """Return ``True`` iff ``stmt`` is ``super().__init__()`` at statement level.

    Strict: zero positional args, zero keywords, both for the
    ``super(...)`` call (must be bare ``super()``) and the
    ``.__init__(...)`` call. Anything richer falls through to the
    validator, which refuses non-trivial super calls.
    """
    if not isinstance(stmt, ast.Expr):
        return False
    call = stmt.value
    if not isinstance(call, ast.Call) or call.args or call.keywords:
        return False
    if not isinstance(call.func, ast.Attribute) or call.func.attr != "__init__":
        return False
    inner = call.func.value
    return (
        isinstance(inner, ast.Call)
        and isinstance(inner.func, ast.Name)
        and inner.func.id == "super"
        and not inner.args
        and not inner.keywords
    )


def _target_name_chain(node: ast.AST) -> str | None:
    """Return the dotted form of a ``Name``/``Attribute`` chain, else ``None``.

    Used to enforce the MVP rule that ``Thread(target=...)`` resolves to
    a statically-known function. ``Name('f')`` returns ``"f"``;
    ``Attribute(Name('m'), 'f')`` returns ``"m.f"``; anything else
    (lambdas, calls, subscripts, arbitrary expressions) returns ``None``.
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


def _scope_bodies(tree: ast.Module) -> list[list[ast.stmt]]:
    """Return the statement list of every Python scope in ``tree``.

    A "scope" is a region with its own local-name binding rules: the
    module top level and the body of every (sync or async) function
    definition. Class bodies and lambdas are intentionally excluded:
    Thread variables cannot meaningfully live in either under the MVP.
    """
    bodies: list[list[ast.stmt]] = [tree.body]
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            bodies.append(node.body)
    return bodies


def _collect_scope_statements(body: list[ast.stmt]) -> list[ast.stmt]:
    """Yield every statement in a scope, descending control-flow constructs.

    Descends into ``If``/``For``/``While``/``With``/``Try`` and their
    ``orelse``/``finalbody``/handler bodies (those share the enclosing
    scope's name bindings) but stops at nested ``FunctionDef`` /
    ``AsyncFunctionDef`` / ``ClassDef`` — those introduce new scopes that
    the caller already handles via :func:`_scope_bodies`.
    """
    out: list[ast.stmt] = []
    for stmt in body:
        out.append(stmt)
        if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            continue
        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            out.extend(_collect_scope_statements(stmt.body))
            out.extend(_collect_scope_statements(stmt.orelse))
        elif isinstance(stmt, ast.If):
            out.extend(_collect_scope_statements(stmt.body))
            out.extend(_collect_scope_statements(stmt.orelse))
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            out.extend(_collect_scope_statements(stmt.body))
        elif isinstance(stmt, ast.Try):
            out.extend(_collect_scope_statements(stmt.body))
            for handler in stmt.handlers:
                out.extend(_collect_scope_statements(handler.body))
            out.extend(_collect_scope_statements(stmt.orelse))
            out.extend(_collect_scope_statements(stmt.finalbody))
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                out.extend(_collect_scope_statements(case.body))
    return out


def _parent_map(tree: ast.AST) -> dict[int, ast.AST]:
    """Build a child-to-parent map for ``tree`` keyed by ``id(node)``.

    ``ast`` does not expose parent pointers; the validator needs them to
    answer "is this Thread construction lexically inside a loop body
    within its own scope" without walking the whole tree per call site.
    """
    parents: dict[int, ast.AST] = {}
    for parent in ast.walk(tree):
        for child in ast.iter_child_nodes(parent):
            parents[id(child)] = parent
    return parents


def _inside_loop_within_scope(node: ast.AST, parents: dict[int, ast.AST]) -> bool:
    """Return ``True`` iff ``node`` is inside a loop body in its own scope.

    Walks the ``parents`` chain rooted at ``node`` and stops at a scope
    boundary (``FunctionDef``/``AsyncFunctionDef``/``Lambda``/``ClassDef``
    /``Module``). Loop ancestors are ``For``/``AsyncFor``/``While`` plus
    the comprehension forms (``ListComp``/``SetComp``/``DictComp``/
    ``GeneratorExp``) — each re-evaluates the node per iteration, which
    the per-site converter state cannot represent.
    """
    cursor = parents.get(id(node))
    while cursor is not None:
        if isinstance(
            cursor,
            (
                ast.FunctionDef,
                ast.AsyncFunctionDef,
                ast.Lambda,
                ast.ClassDef,
                ast.Module,
            ),
        ):
            return False
        if isinstance(
            cursor,
            (
                ast.For,
                ast.AsyncFor,
                ast.While,
                ast.ListComp,
                ast.SetComp,
                ast.DictComp,
                ast.GeneratorExp,
            ),
        ):
            return True
        cursor = parents.get(id(cursor))
    return False


def _scope_name_assign_counts(body: list[ast.stmt]) -> dict[str, int]:
    """Count direct Name-target value bindings per name within a single scope.

    Counts every ``Assign``, ``AnnAssign`` *with* a value, and ``AugAssign``
    whose target is a plain ``Name``, descending into control-flow bodies
    but not into nested defs. ``AnnAssign`` without a value (e.g.
    ``t: Thread``) is intentionally skipped: Python evaluates the
    annotation but does not bind a value, so it is not a definition.

    The result drives the single-definition check that lets the converter
    bind each Thread variable to a unique site id without runtime
    resolution. Rebinds via ``def t(...)``, ``class t``, ``import t``,
    ``for t in ...``, ``with ... as t``, and walrus expressions are not
    counted — those are acceptable MVP gaps documented in
    :func:`validate_threading_thread_usage`.
    """
    counts: dict[str, int] = {}
    for stmt in _collect_scope_statements(body):
        targets: list[ast.expr] = []
        if isinstance(stmt, ast.Assign):
            targets = list(stmt.targets)
        elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
            targets = [stmt.target]
        elif isinstance(stmt, ast.AugAssign):
            targets = [stmt.target]
        for target in targets:
            if isinstance(target, ast.Name):
                counts[target.id] = counts.get(target.id, 0) + 1
    return counts


# pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
def validate_threading_thread_usage(tree: ast.Module, source_filename: str) -> None:
    """Refuse ``threading.Thread`` usage outside the MVP-supported shape.

    ESBMC's Thread lowering (tracked in #4568) is intentionally narrow so
    no concurrency model can be silently weakened. This pass refuses,
    with a precise error message and a non-zero exit, every construct
    outside that shape:

      * subclassing ``threading.Thread``;
      * positional arguments to ``Thread()`` (real signature is
        ``Thread(group, target, name, args, ...)`` and ESBMC only
        accepts the keyword form);
      * ``**kwargs`` splats and ``daemon=``/``name=``/``kwargs=``/
        ``group=`` keyword arguments;
      * ``Thread()`` constructions with no ``target=`` keyword;
      * ``target=`` values that are not a ``Name`` or dotted attribute
        chain (lambdas, calls, runtime variables);
      * ``args=`` values that are not a tuple literal (lists, sets,
        runtime variables);
      * ``Thread()`` constructions lexically inside a loop body within
        their own scope (per-site converter state would be aliased);
      * reassignment of a name that already binds a ``Thread`` instance
        within the same scope (breaks the single-site binding the
        ``.start()`` / ``.join()`` lowering relies on).

    Structurally-valid ``Thread()`` constructions are passed through to
    :func:`lower_threading_thread_usage`, which rewrites them into
    module-level globals, a per-site Python trampoline, and calls to
    the C intrinsics ``__ESBMC_spawn_thread`` / ``__pyt_join`` so
    symex's interleaving search can explore real concurrent behaviour.

    Known gaps (acceptable for the MVP — soundness is preserved because
    each is either over-rejected by the structural fallback or has no
    silent failure path):

      * a ``target=`` bound to a runtime variable holding a callable is
        syntactically a ``Name`` and passes ``_target_name_chain`` —
        AST-only validation cannot distinguish a function name from a
        variable-bound callable without a name-resolution pass. The
        lowering generates a direct call by name; a mismatch surfaces
        as an undefined-symbol error at conversion time, not silent
        weakening of the concurrency model;
      * tuple-target assigns (``a, b = Thread(...), Thread(...)``) and
        walrus-bound Threads are not counted by the reassignment check;
      * rebinds via ``def``/``class``/``import``/``for``-target/
        ``with``-target are not counted by the reassignment check;
      * Thread-subclass name collision with a function-local class of
        the same name — the C2 walk matches subclass constructors by
        bare ``func.id``, so a function-local ``class Worker:`` that
        shadows the module-top ``class Worker(threading.Thread)`` has
        its ``Worker(...)`` calls wrongly refused. This is a false
        positive (valid code rejected), not a false verdict; the user
        can rename the local class. Resolving it would require a
        name-resolution pass.
    """
    module_aliases, thread_aliases = _collect_thread_aliases(tree)
    if not module_aliases and not thread_aliases:
        return

    def fail(line: int, message: str) -> None:
        print(f"ERROR: {source_filename}:{line}: {message}")
        sys.exit(4)

    # Refuse ``from threading import Thread [as X]`` — the Python
    # frontend does not currently resolve such aliases to the
    # operational-model ``Thread`` class, and a downstream
    # segfault/undefined-symbol leak would silently weaken the model.
    # Tracked as an MVP-limited follow-up in #4568.
    if thread_aliases:
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "threading":
                for alias in node.names:
                    if alias.name == "Thread":
                        fail(
                            node.lineno,
                            "`from threading import Thread` is not supported. "
                            "Use `import threading` and reference "
                            "`threading.Thread(...)` directly.",
                        )

    def base_is_thread(base: ast.expr) -> bool:
        return _base_is_thread(base, module_aliases, thread_aliases)

    # MVP subclass support. Collect direct-subclass class defs at module
    # scope; validate each shape (C3-C7); refuse nested subclass defs
    # (C1) so they cannot slip past the module-top collector and reach
    # the converter unstripped.
    subclass_defs = _collect_thread_subclass_defs(
        tree, module_aliases, thread_aliases
    )
    module_top_class_ids = {
        id(node) for node in (tree.body if isinstance(tree, ast.Module) else [])
        if isinstance(node, ast.ClassDef)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(base_is_thread(b) for b in node.bases):
            continue
        if id(node) not in module_top_class_ids:
            fail(
                node.lineno,
                f"`class {node.name}(threading.Thread)` must be defined at "
                "module scope. Nested subclasses are not supported.",
            )

    for cls_name, cls in subclass_defs.items():
        # C7: no multiple inheritance. We strip the Thread base in
        # lowering, but the remaining base(s) would form an MRO the
        # frontend cannot resolve relative to the un-stripped source.
        non_thread_bases = [b for b in cls.bases if not base_is_thread(b)]
        if non_thread_bases:
            fail(
                cls.lineno,
                f"Multiple inheritance is not supported for Thread "
                f"subclass `{cls_name}`. Only "
                f"`class {cls_name}(threading.Thread)` is accepted.",
            )

        # C3: must define run.
        if not any(
            isinstance(m, ast.FunctionDef) and m.name == "run" for m in cls.body
        ):
            fail(
                cls.lineno,
                f"Thread subclass `{cls_name}` must define a `run` method.",
            )

        # C4: cannot override start or join. The lowering replaces
        # calls to these; a user override would be silently bypassed.
        for m in cls.body:
            if (
                isinstance(m, ast.FunctionDef)
                and m.name in THREAD_OVERRIDE_REJECTED_METHODS
            ):
                fail(
                    m.lineno,
                    f"Thread subclass `{cls_name}` overrides `{m.name}`. "
                    "The lowering replaces calls to `start`/`join`; "
                    "overrides would be silently bypassed. Not supported.",
                )

        # C5: super calls inside the subclass body limited to bare
        # ``super().__init__()`` at statement level inside ``__init__``.
        # The lowering strips that one shape; anything richer must be
        # refused so we never silently drop user code.
        for m in cls.body:
            if not isinstance(m, ast.FunctionDef):
                continue
            for inner in ast.walk(m):
                if not _is_super_call_expr(inner):
                    continue
                if m.name == "__init__":
                    # Allow only when the call appears as a bare
                    # ``Expr(super().__init__())`` statement in the
                    # immediate __init__ body — i.e. not nested in an
                    # if/while/with/assignment expression where the
                    # strip pass cannot replace it with Pass.
                    matched = any(
                        _is_super_init_call_stmt(stmt) and stmt.value is inner
                        for stmt in m.body
                    )
                    if matched:
                        continue
                fail(
                    getattr(inner, "lineno", m.lineno),
                    f"Only bare `super().__init__()` at statement level "
                    f"inside `__init__` is supported in Thread subclass "
                    f"`{cls_name}`. Other `super()` usages are not modelled.",
                )

    # Single pass over module top to discharge C2 (module-scope binding)
    # and C6 (single-definition + no rebinding to a non-subclass value).
    # We collect the call ids of subclass constructors found in a valid
    # module-top binding, then refuse any other subclass constructor
    # call. Var-name tracking gives C6's two failure modes inline.
    subclass_names: set[str] = set(subclass_defs)
    module_top_subclass_binding_call_ids: set[int] = set()
    subclass_var_names: set[str] = set()
    if subclass_names and isinstance(tree, ast.Module):
        for stmt in tree.body:
            target_name, value = _extract_name_binding(stmt)
            if target_name is None:
                continue
            is_subclass_ctor = (
                isinstance(value, ast.Call)
                and _is_subclass_constructor(value, subclass_names)
            )
            if is_subclass_ctor:
                if target_name in subclass_var_names:
                    fail(
                        stmt.lineno,
                        f"Thread subclass variable `{target_name}` is "
                        "reassigned at module scope. The single-definition "
                        "rule keeps the spawn-site trampoline's read of "
                        f"`{target_name}` unambiguous.",
                    )
                subclass_var_names.add(target_name)
                module_top_subclass_binding_call_ids.add(id(value))
            elif target_name in subclass_var_names:
                fail(
                    stmt.lineno,
                    f"Thread subclass variable `{target_name}` is rebound "
                    "to a non-subclass value after construction. The "
                    "spawned trampoline would observe the rebound value "
                    "rather than the subclass instance.",
                )

    # C2: every subclass constructor in the tree must be one of the
    # bindings we just registered.
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not _is_subclass_constructor(node, subclass_names):
            continue
        if id(node) not in module_top_subclass_binding_call_ids:
            fail(
                node.lineno,
                f"Thread subclass `{_call_class_name(node)}` must be "
                "constructed at module scope via a simple `<name> = "
                f"{_call_class_name(node)}(...)` binding. Function-scope "
                "bindings, temporaries, and nested expressions are not "
                "supported in the MVP.",
            )

    # Reject Thread construction at class-body scope. The site collector
    # in lower_threading_thread_usage only walks module top + function
    # bodies, so a class-attribute Thread would slip past the rewrite
    # and reach the skeleton's no-kwargs ``__init__``, producing a
    # confusing constructor-arity error rather than a clear MVP message.
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for stmt in node.body:
            _, value = _extract_name_binding(stmt)
            if isinstance(value, ast.Call) and _is_thread_constructor(
                value, module_aliases, thread_aliases
            ):
                fail(
                    stmt.lineno,
                    "threading.Thread bound at class-attribute scope is "
                    "not supported. Bind the Thread in __init__ or at "
                    "module/function scope.",
                )

    parents = _parent_map(tree)

    # Reject Thread() construction inside loop bodies in the same scope.
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and _is_thread_constructor(
            node, module_aliases, thread_aliases
        ) and _inside_loop_within_scope(node, parents):
            fail(
                node.lineno,
                "threading.Thread construction inside a loop is not yet "
                "supported. Construct each Thread at a distinct top-level "
                "or function-scope site.",
            )

    # Per-construction-site structural checks.
    for call_node in ast.walk(tree):
        if not isinstance(call_node, ast.Call):
            continue
        if not _is_thread_constructor(call_node, module_aliases, thread_aliases):
            continue

        if call_node.args:
            fail(
                call_node.lineno,
                "threading.Thread requires `target=` and (optionally) "
                "`args=` as keyword arguments; positional arguments are "
                "not supported.",
            )

        target_value: ast.expr | None = None
        args_value: ast.expr | None = None
        for kw in call_node.keywords:
            if kw.arg is None:
                fail(
                    call_node.lineno,
                    "threading.Thread(**kwargs) is not supported; use the "
                    "`target=...` and `args=...` keyword arguments directly.",
                )
            if kw.arg in UNSUPPORTED_THREAD_KWARGS:
                fail(
                    call_node.lineno,
                    f"threading.Thread keyword argument `{kw.arg}=` is not "
                    "supported by ESBMC. Only target and args are modelled.",
                )
            if kw.arg == "target":
                target_value = kw.value
            elif kw.arg == "args":
                args_value = kw.value

        if target_value is None:
            fail(
                call_node.lineno,
                "threading.Thread requires `target=<function>`; "
                "constructions without an explicit target are not supported.",
            )
            return  # unreachable: fail() exits; narrows target_value for type-checkers.

        if _target_name_chain(target_value) is None:
            fail(
                call_node.lineno,
                "threading.Thread `target=` must be a function name (or "
                "attribute chain). Lambdas and runtime-variable callables "
                "are not supported in the MVP.",
            )

        if args_value is not None and not isinstance(args_value, ast.Tuple):
            fail(
                call_node.lineno,
                "threading.Thread `args=` must be a tuple literal "
                "(e.g. `args=(resource,)`). Lists, sets, and runtime "
                "variables are not supported in the MVP.",
            )

    # Per-scope single-definition check.
    for body in _scope_bodies(tree):
        counts = _scope_name_assign_counts(body)
        for stmt in _collect_scope_statements(body):
            target_name, value = _extract_name_binding(stmt)
            if (
                target_name is not None
                and isinstance(value, ast.Call)
                and _is_thread_constructor(value, module_aliases, thread_aliases)
                and counts.get(target_name, 0) > 1
            ):
                fail(
                    stmt.lineno,
                    f"threading.Thread variable `{target_name}` is "
                    "reassigned in the same scope. The Thread model "
                    "requires a single-definition binding so the spawn "
                    "site can resolve the target statically.",
                )



def _thread_method_receiver(
    node: ast.AST,
    thread_var_names: set[str],
    method: str,
) -> str | None:
    """Return the receiver var name iff ``node`` is ``var.<method>()`` on a tracked Thread.

    Recognises ``Name(var).method()`` shape only; attribute-chain
    receivers (``obj.t.start()``) are out of MVP scope. Returning the
    name directly (rather than a bool) lets the caller skip a separate
    type-narrowing pass on ``call.func``.
    """
    if not isinstance(node, ast.Call):
        return None
    func = node.func
    if not isinstance(func, ast.Attribute) or func.attr != method:
        return None
    if isinstance(func.value, ast.Name) and func.value.id in thread_var_names:
        return func.value.id
    return None


def _collect_thread_var_sites(
    tree: ast.Module,
    module_aliases: set[str],
    thread_aliases: set[str],
    starting_id: int = 0,
) -> tuple[dict[int, dict[str, tuple[int, ast.Call]]], int]:
    """Return ``({id(scope_body): {var_name: (site_id, construction_call)}}, next_id)``.

    Each scope (module top, each function body) gets its own inner map
    because the validator's single-definition check is per-scope: the
    same name ``t`` may legally bind a Thread in two distinct function
    scopes, and each binding must allocate its own site id (otherwise
    the second scope's ``t.start()`` would silently reuse the first
    scope's trampoline). Site ids are assigned in deterministic source
    order so the generated globals / trampolines stay stable across
    parser runs.

    ``starting_id`` lets the caller chain this collector with the
    subclass-site collector under one continuous id space.
    """
    sites: dict[int, dict[str, tuple[int, ast.Call]]] = {}
    next_id = starting_id
    for body in _scope_bodies(tree):
        scope_sites: dict[str, tuple[int, ast.Call]] = {}
        for stmt in _collect_scope_statements(body):
            target_name, value = _extract_name_binding(stmt)
            if (
                target_name is not None
                and target_name not in scope_sites
                and isinstance(value, ast.Call)
                and _is_thread_constructor(value, module_aliases, thread_aliases)
            ):
                scope_sites[target_name] = (next_id, value)
                next_id += 1
        if scope_sites:
            sites[id(body)] = scope_sites
    return sites, next_id


def _collect_subclass_sites(
    tree: ast.Module,
    subclass_names: set[str],
    starting_id: int,
) -> tuple[dict[str, tuple[int, ast.Call]], int]:
    """Return ``({var_name: (site_id, call_node)}, next_id)`` for module-top subclass bindings.

    Validator C2 guarantees subclass instance bindings are module-scope
    only, so we walk ``tree.body`` (not the per-scope helper) and the
    returned map is flat — every entry is a module-global var name
    visible to every other scope. Site ids continue from
    ``starting_id`` so target= and subclass sites share one id space
    and the generated trampoline / tid names never collide.
    """
    out: dict[str, tuple[int, ast.Call]] = {}
    next_id = starting_id
    if not subclass_names or not isinstance(tree, ast.Module):
        return out, next_id
    for stmt in tree.body:
        target_name, value = _extract_name_binding(stmt)
        if (
            target_name is not None
            and target_name not in out
            and isinstance(value, ast.Call)
            and _is_subclass_constructor(value, subclass_names)
        ):
            out[target_name] = (next_id, value)
            next_id += 1
    return out, next_id


def _thread_call_keywords(call_node: ast.Call) -> tuple[ast.expr, list[ast.expr]]:
    """Return ``(target_expr, list_of_arg_exprs)`` for a validated Thread call.

    ``args=`` (when omitted) yields an empty arg list. The validator has
    already proven ``target=`` is present and ``args=`` (if present) is
    a tuple literal, so this function only does the lookup.
    """
    target_value: ast.expr | None = None
    args_values: list[ast.expr] = []
    for kw in call_node.keywords:
        # Defence-in-depth: the validator already refuses anything other
        # than ``target=`` and ``args=`` (daemon/name/kwargs/**splat are
        # rejected upstream). If a fresh validator gap ever lets one
        # through, fail loudly here rather than silently dropping it.
        # Use ``raise`` rather than ``assert`` so the check survives
        # ``python -O``.
        if kw.arg not in ("target", "args"):
            raise RuntimeError(
                f"unexpected Thread() kwarg {kw.arg!r} reached lowering; "
                "validator gap"
            )
        if kw.arg == "target":
            target_value = kw.value
        elif kw.arg == "args" and isinstance(kw.value, ast.Tuple):
            args_values = list(kw.value.elts)
    if target_value is None:
        # validator guarantees ``target=`` is present
        raise RuntimeError(
            "Thread() construction reached lowering without target= kwarg; "
            "validator gap"
        )
    return target_value, args_values


def _build_target_call(target_expr: ast.expr, site_id: int, n_args: int) -> ast.expr:
    """Return an AST for ``<target>(<arg0>, ..., <argN-1>)``.

    ``<argI>`` is the module-level global ``__pythread_arg_<site>_<i>``
    populated at construction time. ``<target>`` is a deep copy of the
    original ``target=`` chain so the trampoline calls the same function
    the user named.
    """
    args: list[ast.expr] = [
        ast.Name(id=f"__pythread_arg_{site_id}_{i}", ctx=ast.Load())
        for i in range(n_args)
    ]
    return ast.Call(func=copy.deepcopy(target_expr), args=args, keywords=[])


def _build_trampoline(site_id: int, target_expr: ast.expr, n_args: int) -> ast.FunctionDef:
    """Synthesise the per-site Python trampoline ``FunctionDef``.

    Body is::

        target(__pythread_arg_<N>_0, ..., __pythread_arg_<N>_n-1)
        __pyt_terminate()

    The trampoline is invoked by ``__ESBMC_spawn_thread`` as a spawned
    thread; ``__pyt_terminate`` marks the thread ended so ``__pyt_join``
    can observe completion. The leading ``global`` declaration is what
    lets the trampoline read the construction-site arg globals — without
    it Python's name resolution treats every ``__pythread_arg_<N>_<i>``
    as a local lookup, which the ESBMC frontend rejects with an
    undefined-variable error.
    """
    body: list[ast.stmt] = []
    if n_args:
        body.append(
            ast.Global(
                names=[f"__pythread_arg_{site_id}_{i}" for i in range(n_args)]
            )
        )
    body.extend(
        [
            ast.Expr(value=_build_target_call(target_expr, site_id, n_args)),
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="__pyt_terminate", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
            ),
        ]
    )
    fn = ast.FunctionDef(
        name=f"__pythread_trampoline_{site_id}",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=ast.Constant(value=None),
    )
    return fn


def _build_subclass_trampoline(site_id: int, var_name: str) -> ast.FunctionDef:
    """Synthesise the trampoline for a Thread subclass instance bound to ``var_name``.

    Body is::

        global <var_name>
        <var_name>.run()
        __pyt_terminate()

    The ``global`` lets the trampoline read the user's module-level
    instance binding (validator C2 guarantees module scope). The
    trampoline must be inserted *after* the user's ``<var_name> = X(...)``
    statement so the Python frontend resolves the type of ``<var_name>``
    by the time it sees the trampoline body.
    """
    body: list[ast.stmt] = [
        ast.Global(names=[var_name]),
        ast.Expr(
            value=ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=var_name, ctx=ast.Load()),
                    attr="run",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        ),
        ast.Expr(
            value=ast.Call(
                func=ast.Name(id="__pyt_terminate", ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        ),
    ]
    return ast.FunctionDef(
        name=f"__pythread_trampoline_{site_id}",
        args=ast.arguments(
            posonlyargs=[],
            args=[],
            vararg=None,
            kwonlyargs=[],
            kw_defaults=[],
            kwarg=None,
            defaults=[],
        ),
        body=body,
        decorator_list=[],
        returns=ast.Constant(value=None),
    )


def _build_tid_declaration(site_id: int) -> ast.Assign:
    """Return ``__pythread_tid_<N> = 0`` as a module-top assignment."""
    return ast.Assign(
        targets=[ast.Name(id=f"__pythread_tid_{site_id}", ctx=ast.Store())],
        value=ast.Constant(value=0),
    )


def _strip_thread_inheritance(
    class_node: ast.ClassDef,
    module_aliases: set[str],
    thread_aliases: set[str],
) -> None:
    """Remove ``threading.Thread`` from ``class_node.bases`` and neuter ``super().__init__()``.

    Two in-place transforms required to make the user's subclass
    survive the Python frontend's class converter:

      * Strip every base that resolves to ``threading.Thread``. The
        converter cannot resolve the operational-model ``Thread``
        skeleton as a base (probe: ``nlohmann::json`` key-not-found
        assertion). After stripping, ``class X(threading.Thread)``
        becomes ``class X:`` (implicit ``object`` base).
      * Replace any bare ``super().__init__()`` statement inside
        ``__init__`` with ``Pass``. The converter cannot resolve
        ``super().__init__()`` on an implicit ``object`` base either
        (probe: ``_init_undefined`` / ``migrate expr failed``). The
        validator (C5) refuses every other ``super(...)`` shape, so
        we only need to neutralise the bare-init shape here.
    """
    class_node.bases = [
        b for b in class_node.bases
        if not _base_is_thread(b, module_aliases, thread_aliases)
    ]
    for stmt in class_node.body:
        if not (isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__"):
            continue
        stmt.body = [
            ast.Pass() if _is_super_init_call_stmt(inner) else inner
            for inner in stmt.body
        ]


def _build_arg_declaration(
    site_id: int,
    arg_index: int,
    arg_value: ast.expr,
) -> ast.stmt:
    """Return a module-top declaration for ``__pythread_arg_<N>_<i>``.

    A plain ``Assign`` or an ``AnnAssign`` with a non-null ``value`` is
    required so the ESBMC Python frontend creates a *module-level*
    binding: an annotation-only declaration silently degrades to a
    function-local at the first construction-site write, and the
    spawned trampoline would then read an uninitialised global rather
    than the user-supplied argument.

    The declared shape is chosen by ``arg_value`` (the construction-
    site expression, treated read-only):

      * Numeric / string constants — emit ``= <zero-of-that-kind>`` so
        the symbol's inferred type matches the construction-site rebind
        and the trampoline call site type-checks against the target's
        parameter.
      * Anything else (Name references, attribute chains, calls,
        list/dict/set literals, ``None``) — emit ``: object = None``.
        ``object`` lowers to ``any_type()`` (``void *``) in the Python
        frontend, so a class-instance arg carrying a struct pointer is
        forwarded to the trampoline call site without the
        int-degradation that ``= 0`` would force.
    """
    target = ast.Name(
        id=f"__pythread_arg_{site_id}_{arg_index}", ctx=ast.Store()
    )

    if isinstance(arg_value, ast.Constant):
        v = arg_value.value
        # bool must be checked before int: isinstance(True, int) is True.
        if isinstance(v, bool):
            return ast.Assign(targets=[target], value=ast.Constant(value=False))
        if isinstance(v, int):
            return ast.Assign(targets=[target], value=ast.Constant(value=0))
        if isinstance(v, float):
            return ast.Assign(targets=[target], value=ast.Constant(value=0.0))
        if isinstance(v, str):
            return ast.Assign(targets=[target], value=ast.Constant(value=""))
        if isinstance(v, bytes):
            return ast.Assign(targets=[target], value=ast.Constant(value=b""))
        # ``None`` and other constant kinds (Ellipsis, complex) fall
        # through to the object-typed declaration below.

    return ast.AnnAssign(
        target=target,
        annotation=ast.Name(id="object", ctx=ast.Load()),
        value=ast.Constant(value=None),
        simple=1,
    )


def _build_arg_assignment(site_id: int, arg_index: int, value: ast.expr) -> ast.Assign:
    """Return ``__pythread_arg_<N>_<i> = <value>`` as an AST node.

    The deep-copied ``value`` is the construction-site expression; the
    assignment runs in the user's calling frame just before the
    rewritten bare ``Thread()`` construction so module globals carry
    the construction-time value into the spawned thread.
    """
    return ast.Assign(
        targets=[
            ast.Name(id=f"__pythread_arg_{site_id}_{arg_index}", ctx=ast.Store())
        ],
        value=copy.deepcopy(value),
    )


def _build_bare_thread_call(original_call: ast.Call) -> ast.Call:
    """Return a no-kwargs ``Thread()`` call mirroring ``original_call.func``.

    Preserves the original constructor reference (``threading.Thread``
    vs ``MyThread``) so the Python frontend's class resolution still
    finds the skeleton, but strips the ``target=`` / ``args=`` keywords
    so the skeleton's zero-arg ``__init__`` accepts the call.
    """
    return ast.Call(func=copy.deepcopy(original_call.func), args=[], keywords=[])


def _build_start_statements(site_id: int) -> list[ast.stmt]:
    """Return the atomic spawn-and-init sequence replacing ``t.start()``.

    Emits, in order:

      1. ``global __pythread_tid_<N>`` — binds the tid assignment to
         the module-level slot the prelude declared, even when
         ``t.start()`` is called inside a function scope.
      2. ``__ESBMC_atomic_begin()``.
      3. ``__pythread_tid_<N> = __ESBMC_spawn_thread(__pythread_trampoline_<N>)``.
      4. ``__pyt_init_tid(__pythread_tid_<N>)``.
      5. ``__ESBMC_atomic_end()``.

    The atomic block is essential: without it, the spawned trampoline
    can run (and reach ``__pyt_terminate``, setting
    ``pthread_thread_ended[tid] = 1``) between the spawn and the init,
    after which ``__pyt_init_tid`` would reset the ended flag back to
    0 and the subsequent ``__pyt_join`` would deadlock-falsely.
    """
    def call(name: str, args: list[ast.expr]) -> ast.Call:
        return ast.Call(
            func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[]
        )

    tramp_arg = ast.Name(id=f"__pythread_trampoline_{site_id}", ctx=ast.Load())
    tid_arg = ast.Name(id=f"__pythread_tid_{site_id}", ctx=ast.Load())
    tid_target = ast.Name(id=f"__pythread_tid_{site_id}", ctx=ast.Store())
    return [
        ast.Global(names=[f"__pythread_tid_{site_id}"]),
        ast.Expr(value=call("__ESBMC_atomic_begin", [])),
        ast.Assign(
            targets=[tid_target],
            value=call("__ESBMC_spawn_thread", [tramp_arg]),
        ),
        ast.Expr(value=call("__pyt_init_tid", [tid_arg])),
        ast.Expr(value=call("__ESBMC_atomic_end", [])),
    ]


def _build_join_call(site_id: int) -> ast.Expr:
    """Return ``__pyt_join(__pythread_tid_<N>)`` as a statement."""
    return ast.Expr(
        value=ast.Call(
            func=ast.Name(id="__pyt_join", ctx=ast.Load()),
            args=[ast.Name(id=f"__pythread_tid_{site_id}", ctx=ast.Load())],
            keywords=[],
        )
    )


def _rewrite_construction_stmt(
    stmt: ast.stmt,
    call_node: ast.Call,
    site_id: int,
) -> list[ast.stmt]:
    """Return the replacement statements for a Thread construction.

    The original ``t = Thread(target=f, args=(x, y))`` becomes::

        global __pythread_arg_<N>_0, __pythread_arg_<N>_1
        __pythread_arg_<N>_0 = x
        __pythread_arg_<N>_1 = y
        t = Thread()

    The leading ``global`` ensures the arg writes bind to the module-
    level placeholders the prelude declared, not to function locals
    that the spawned trampoline could never see. (For module-scope
    constructions the ``global`` is redundant but harmless.) The spawn
    happens later at ``t.start()``, by which time the args are visible
    to the spawned trampoline (cross-thread visibility follows from
    the spawn happens-before edge).
    """
    _, args_values = _thread_call_keywords(call_node)
    out: list[ast.stmt] = []
    if args_values:
        out.append(
            ast.Global(
                names=[
                    f"__pythread_arg_{site_id}_{i}"
                    for i in range(len(args_values))
                ]
            )
        )
    for i, arg_value in enumerate(args_values):
        out.append(_build_arg_assignment(site_id, i, arg_value))
    bare = _build_bare_thread_call(call_node)
    if isinstance(stmt, ast.Assign):
        out.append(ast.Assign(targets=stmt.targets, value=bare))
    elif isinstance(stmt, ast.AnnAssign):
        out.append(
            ast.AnnAssign(
                target=stmt.target,
                annotation=stmt.annotation,
                value=bare,
                simple=stmt.simple,
            )
        )
    else:
        # _try_rewrite_statement only dispatches Assign/AnnAssign here.
        raise RuntimeError(
            f"_rewrite_construction_stmt received unexpected stmt type "
            f"{type(stmt).__name__}"
        )
    return out


# pylint: disable-next=too-many-locals,too-many-branches
def lower_threading_thread_usage(
    tree: ast.Module, source_filename: str
) -> None:
    """Rewrite ``threading.Thread`` usage into pthread-backed intrinsics.

    Runs after :func:`validate_threading_thread_usage` has refused every
    unsupported shape. For each accepted ``Thread(...)`` construction
    site:

      * allocate a unique site id;
      * emit module-level globals ``__pythread_arg_<N>_<i>`` for each
        ``args=`` element and a ``__pythread_tid_<N>`` slot for the
        spawned thread's id;
      * emit a module-level trampoline ``__pythread_trampoline_<N>``
        whose body calls the user-supplied target with the per-site
        arg globals, then calls ``__pyt_terminate``;
      * rewrite the construction statement to populate the arg globals
        and bind the user's name to a bare ``Thread()``;
      * rewrite every ``t.start()`` call to an atomic block that
        captures the spawned tid into ``__pythread_tid_<N>`` and calls
        ``__pyt_init_tid``, then rewrite every ``t.join()`` call to
        ``__pyt_join(__pythread_tid_<N>)``.

    Producing module-level globals (rather than function-locals
    captured by closure) was the fix for #4571's three soundness
    defects: it gives the spawned trampoline an unambiguous reader of
    the construction-site values, and ``__pyt_join`` provides
    the real synchronisation edge symex's interleaving search needs.
    """
    module_aliases, thread_aliases = _collect_thread_aliases(tree)
    if not module_aliases and not thread_aliases:
        return

    # Stage 1: strip ``threading.Thread`` base and ``super().__init__()``
    # from every direct subclass at module scope. Must happen before
    # site collection so the per-site post-processing sees the rewritten
    # class def.
    subclass_defs = _collect_thread_subclass_defs(
        tree, module_aliases, thread_aliases
    )
    for cls in subclass_defs.values():
        _strip_thread_inheritance(cls, module_aliases, thread_aliases)

    # Stage 2: collect both site shapes under one continuous site-id
    # space so the generated trampoline / tid names never collide.
    sites_by_scope, next_id = _collect_thread_var_sites(
        tree, module_aliases, thread_aliases
    )
    subclass_sites, next_id = _collect_subclass_sites(
        tree, set(subclass_defs), next_id
    )

    if not sites_by_scope and not subclass_sites:
        return

    # Build the prelude: arg globals (zero-initialised so the Python
    # frontend's name resolver sees a module-level binding when the
    # trampoline references them), tid slot, and trampoline. One set
    # per site_id, deterministic source order.
    prelude: list[ast.stmt] = []
    for scope_sites in sites_by_scope.values():
        for _, (site_id, call_node) in scope_sites.items():
            target_expr, args_values = _thread_call_keywords(call_node)
            for i, arg_value in enumerate(args_values):
                prelude.append(
                    _build_arg_declaration(site_id, i, arg_value)
                )
            prelude.append(_build_tid_declaration(site_id))
            prelude.append(
                _build_trampoline(site_id, target_expr, len(args_values))
            )

    # Pick the insertion point that satisfies two source-order
    # constraints simultaneously:
    #
    #   * trampoline bodies reference the user's target function
    #     (e.g. ``setter(__pythread_arg_0_0)``); the target must be
    #     defined earlier so PR #4570's call-site parameter-type
    #     inference can match call args to the target's params.
    #   * the rewritten ``__ESBMC_spawn_thread(__pythread_trampoline_<N>)``
    #     references the trampoline as a Name-as-argument; the Python
    #     frontend resolves such references eagerly at conversion
    #     time, so the trampoline FunctionDef must precede the spawn
    #     site.
    #
    # Concretely: ``insert_at`` must be > the index of the latest
    # top-level def whose name is referenced as a ``target=`` (constraint
    # 1) AND <= the index of the earliest top-level function whose body
    # contains a Thread construction (constraint 2). If those bounds
    # invert — caller function defined before the target function it
    # references — the user's code cannot be safely lowered; fail loud
    # rather than emit a malformed prelude.
    inner_thread_scopes = {
        id(node.body)
        for node in ast.walk(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and id(node.body) in sites_by_scope
    }
    target_names: set[str] = set()
    for scope_sites in sites_by_scope.values():
        for _, call_node in scope_sites.values():
            target_expr, _ = _thread_call_keywords(call_node)
            leftmost = target_expr
            while isinstance(leftmost, ast.Attribute):
                leftmost = leftmost.value
            if isinstance(leftmost, ast.Name):
                target_names.add(leftmost.id)
    insert_at = 0
    latest_target_def_idx = -1
    earliest_user_with_thread: int | None = None
    for idx, stmt in enumerate(tree.body):
        if isinstance(
            stmt,
            (
                ast.Import,
                ast.ImportFrom,
                ast.ClassDef,
                ast.FunctionDef,
                ast.AsyncFunctionDef,
            ),
        ):
            insert_at = idx + 1
        if (
            isinstance(
                stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
            )
            and stmt.name in target_names
        ):
            latest_target_def_idx = idx
        if (
            earliest_user_with_thread is None
            and isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
            and id(stmt.body) in inner_thread_scopes
        ):
            earliest_user_with_thread = idx
    if earliest_user_with_thread is not None:
        if latest_target_def_idx >= earliest_user_with_thread:
            offender = tree.body[earliest_user_with_thread]
            print(
                f"ERROR: {source_filename}:{offender.lineno}: "
                "threading.Thread target=<name> must be defined before "
                "the function that constructs the Thread; define the "
                "target above its caller, or move the construction to "
                "module scope."
            )
            sys.exit(4)
        insert_at = earliest_user_with_thread
    else:
        insert_at = max(insert_at, latest_target_def_idx + 1)
    if prelude:
        tree.body[insert_at:insert_at] = prelude

    # Subclass per-binding injection: insert ``__pythread_tid_<N> = 0``
    # and ``def __pythread_trampoline_<N>():`` immediately after each
    # construction statement at module top. The trampoline reads the
    # user's instance binding, so it must appear *after* the binding
    # statement in source order. Process in reverse index order so
    # earlier insertions do not shift later ones.
    subclass_insertions: list[tuple[int, list[ast.stmt]]] = []
    for var_name, (site_id, call_node) in subclass_sites.items():
        binding_idx = _find_binding_stmt_index(tree, call_node)
        if binding_idx is None:
            # Validator C2 guarantees module-top binding; reaching this
            # means a validator gap. Fail loudly.
            raise RuntimeError(
                f"subclass binding `{var_name}` reached lowering without "
                f"a module-top binding statement; validator gap"
            )
        subclass_insertions.append(
            (
                binding_idx,
                [
                    _build_tid_declaration(site_id),
                    _build_subclass_trampoline(site_id, var_name),
                ],
            )
        )
    for binding_idx, stmts in sorted(subclass_insertions, reverse=True):
        tree.body[binding_idx + 1:binding_idx + 1] = stmts

    # Rewrite Thread() constructions and start()/join() calls in place.
    # Subclass var names are module-global (validator C2), so they are
    # visible from every scope — merge them into every scope's site map.
    # Use setdefault so a local ``target=`` binding (with the same
    # name as a module-top subclass binding) wins for its own scope:
    # without this precedence rule the subclass entry would silently
    # steal the function-scope start()/join() routing, redirecting them
    # to the subclass trampoline and producing false VERIFICATION
    # SUCCESSFUL on programs whose real spawn point is the local Thread.
    for body in _scope_bodies(tree):
        scope_sites = dict(sites_by_scope.get(id(body), {}))
        for var_name, site_info in subclass_sites.items():
            scope_sites.setdefault(var_name, site_info)
        if scope_sites:
            _rewrite_body_in_place(
                body, scope_sites, module_aliases, thread_aliases
            )

    ast.fix_missing_locations(tree)


def _find_binding_stmt_index(
    tree: ast.Module, call_node: ast.Call
) -> int | None:
    """Return the module-top index of the Assign/AnnAssign whose value is ``call_node``."""
    for idx, stmt in enumerate(tree.body):
        if isinstance(stmt, (ast.Assign, ast.AnnAssign)) and stmt.value is call_node:
            return idx
    return None


def _rewrite_body_in_place(
    body: list[ast.stmt],
    scope_sites: dict[str, tuple[int, ast.Call]],
    module_aliases: set[str],
    thread_aliases: set[str],
) -> None:
    """Walk ``body`` recursively, rewriting this scope's Thread sites.

    ``scope_sites`` is the inner map from :func:`_collect_thread_var_sites`
    for *this* scope only — sibling scopes are visited separately so
    their Thread variables with the same name resolve to distinct site
    ids. Descends into control-flow constructs (``If``/``For``/``While``
    /``With``/``Try``) but not into nested defs (those are independent
    scopes already enrolled in the outer site map).

    ``module_aliases`` / ``thread_aliases`` are forwarded so the
    construction-rewrite predicate in :func:`_try_rewrite_statement`
    can require a real ``threading.Thread`` call rather than misfiring
    on a subclass constructor whose own ``__init__`` happens to take a
    ``target=`` keyword argument.
    """
    i = 0
    while i < len(body):
        stmt = body[i]
        rewritten = _try_rewrite_statement(
            stmt, scope_sites, module_aliases, thread_aliases
        )
        if rewritten is not None:
            body[i:i + 1] = rewritten
            i += len(rewritten)
            continue

        if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While)):
            _rewrite_body_in_place(stmt.body, scope_sites, module_aliases, thread_aliases)
            _rewrite_body_in_place(stmt.orelse, scope_sites, module_aliases, thread_aliases)
        elif isinstance(stmt, ast.If):
            _rewrite_body_in_place(stmt.body, scope_sites, module_aliases, thread_aliases)
            _rewrite_body_in_place(stmt.orelse, scope_sites, module_aliases, thread_aliases)
        elif isinstance(stmt, (ast.With, ast.AsyncWith)):
            _rewrite_body_in_place(stmt.body, scope_sites, module_aliases, thread_aliases)
        elif isinstance(stmt, ast.Try):
            _rewrite_body_in_place(stmt.body, scope_sites, module_aliases, thread_aliases)
            for handler in stmt.handlers:
                _rewrite_body_in_place(handler.body, scope_sites, module_aliases, thread_aliases)
            _rewrite_body_in_place(stmt.orelse, scope_sites, module_aliases, thread_aliases)
            _rewrite_body_in_place(stmt.finalbody, scope_sites, module_aliases, thread_aliases)
        elif isinstance(stmt, ast.Match):
            for case in stmt.cases:
                _rewrite_body_in_place(case.body, scope_sites, module_aliases, thread_aliases)
        i += 1


def _try_rewrite_statement(
    stmt: ast.stmt,
    scope_sites: dict[str, tuple[int, ast.Call]],
    module_aliases: set[str],
    thread_aliases: set[str],
) -> list[ast.stmt] | None:
    """Rewrite a single Thread construction / start / join statement.

    Returns ``None`` if ``stmt`` is unrelated to threading; otherwise
    returns the replacement statement list. ``scope_sites`` is the
    per-scope ``{var_name: (site_id, call_node)}`` map.

    The construction-rewrite arm requires the call to be a real
    ``threading.Thread`` constructor (via :func:`_is_thread_constructor`)
    — *not* merely "has a ``target=`` keyword". A user subclass whose
    ``__init__`` accepts a parameter named ``target`` would otherwise
    have its keyword silently dropped by :func:`_build_bare_thread_call`,
    a soundness defect.
    """
    target_name, value = _extract_name_binding(stmt)
    if (
        target_name is not None
        and target_name in scope_sites
        and _is_thread_construction_with_target_kw(
            value, module_aliases, thread_aliases)
    ):
        return _rewrite_construction_stmt(stmt, value, scope_sites[target_name][0])

    # Method call: t.start() or t.join() at statement level.
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
        call = stmt.value
        thread_var_names = set(scope_sites)
        receiver = _thread_method_receiver(call, thread_var_names, "start")
        if receiver is not None:
            return _build_start_statements(scope_sites[receiver][0])
        receiver = _thread_method_receiver(call, thread_var_names, "join")
        if receiver is not None:
            return [_build_join_call(scope_sites[receiver][0])]
    return None


def is_testing_framework(module_name):
    # Check if module is a testing framework that should be skipped.
    testing_frameworks = [
        "pytest",
    ]
    return module_name in testing_frameworks


def import_module_by_name(module_name, output_dir):
    if is_unsupported_module(module_name):
        print(f"ERROR: \"import {module_name}\" is not supported")
        sys.exit(3)

    base_module = module_name.split(".")[0]

    # Skip testing frameworks - they don't contain logic to verify
    if is_testing_framework(base_module):
        return None
    if is_imported_model(base_module):
        parts = module_name.split(".")
        model_dir = os.path.join(output_dir, "models")
        path = os.path.join(model_dir, *parts) + ".py"

        if not os.path.exists(path):
            path = os.path.join(model_dir, *parts, "__init__.py")

        return os.path.abspath(path)

    try:
        module = importlib.import_module(module_name)
        return module
    except ImportError:
        # Try importing the parent module if this looks like a class/attribute reference
        if "." in module_name:
            parent = ".".join(module_name.split(".")[:-1])
            try:
                return importlib.import_module(parent)
            except ImportError:
                pass

        print(f"ERROR: Module '{module_name}' not found.")
        print(f"Please install it with: pip3 install {module_name}")
        return None


def encode_bytes(value):
    return base64.b64encode(value).decode('ascii')


def annotate_constant_node(value_node):
    # Python 3.8+ uses ast.Constant instead of ast.Str, ast.Num, ast.Bytes, etc.
    if not isinstance(value_node, ast.Constant):
        return

    if isinstance(value_node.value, str):
        value_node.esbmc_type_annotation = "str"
    elif isinstance(value_node.value, bytes):
        value_node.esbmc_type_annotation = "bytes"
        value_node.encoded_bytes = encode_bytes(value_node.value)
    elif isinstance(value_node.value, complex):
        value_node.esbmc_type_annotation = "complex"
        value_node.real_value = value_node.value.real
        value_node.imag_value = value_node.value.imag


def add_type_annotation(node):
    annotate_constant_node(node.value)


def is_standard_library_file(filename):
    stdlib_paths = [
        '/usr/lib/python',
        '/usr/local/lib/python',
        '/Library/Frameworks/Python.framework',
        '/opt/homebrew/Cellar/python',  # Homebrew Python on macOS (Apple Silicon)
        '/usr/local/Cellar/python',  # Homebrew Python on macOS (Intel)
        '/opt/conda/lib/python',  # Conda standard installation path
    ]
    # Check fixed paths first (no expanduser needed)
    if any(filename.startswith(path) for path in stdlib_paths):
        return True
    # Check pyenv paths
    pyenv_root = os.environ.get('PYENV_ROOT', os.path.expanduser('~/.pyenv'))
    if pyenv_root and filename.startswith(pyenv_root):
        # Check if it's in the versions directory (standard library location)
        if '/versions/' in filename and '/lib/python' in filename:
            return True
    # Check conda paths (including user installations)
    if filename.startswith(os.path.expanduser('~/miniconda3/lib/python')) or \
       filename.startswith(os.path.expanduser('~/anaconda3/lib/python')):
        return True
    return False


def expand_star_import(module) -> list[str] | None:
    names = getattr(module, '__all__', None)
    if names is None:
        names = [n for n in dir(module) if not n.startswith('_')]
    return names


def get_referenced_names(node):
    """
    Find all functions and classes referenced in a function or class definition.

    Returns a set of names that are called as functions or used in type annotations.
    """
    referenced = set()

    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            # Check if it's a direct function/class call (simple Name node)
            if isinstance(child.func, ast.Name):
                referenced.add(child.func.id)

        # Check for names in type annotations (return types, argument types, etc.)
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Return type annotation
            if child.returns and isinstance(child.returns, ast.Name):
                referenced.add(child.returns.id)
            # Argument type annotations
            for arg in child.args.args:
                if arg.annotation and isinstance(arg.annotation, ast.Name):
                    referenced.add(arg.annotation.id)

        # Variable annotations (e.g., x: Foo = ...)
        elif isinstance(child, ast.AnnAssign):
            if isinstance(child.annotation, ast.Name):
                referenced.add(child.annotation.id)

    return referenced


import_aliases = {}
# Track all imports per module to combine them
module_imports = {}
# Per-module export tables collected after Preprocessor runs (#4525). Each
# entry is (range_aliases, range_wrappers, dunder_all_or_None) and is keyed
# by qualified module name. The entry script is keyed by ``__main__``.
module_exports = {}


# pylint: disable-next=too-many-locals,too-many-branches
def process_imports(node, output_dir):
    """
    Process import statements in the AST node.

    Parameters
    ----------
    node
        The import node to process.
    output_dir
        The directory to save the generated JSON files.

    """
    imported_elements = None
    module_names = []
    if isinstance(node, (ast.Import)):
        for alias_node in node.names:
            module_name = alias_node.name
            alias = alias_node.asname or module_name
            import_aliases[alias] = module_name
            module_names.append(module_name)
        if not module_names:
            return
    elif isinstance(node, ast.ImportFrom):
        module_name = node.module
        # If it's a star import, leave imported_elements as None to import everything
        if not any(a.name == '*' for a in node.names):
            imported_elements = node.names
        if module_name:
            import_aliases[module_name] = module_name
        module_names = [module_name] if module_name else []
        if not module_names:
            return

    # Track imports for this module
    for module_name in module_names:
        if module_name not in module_imports:
            module_imports[module_name] = {'import_all': False, 'specific_names': set()}

        if imported_elements is None:
            # This is an "import module" or "from module import *"; mark to import everything
            module_imports[module_name]['import_all'] = True
        else:
            # Add specific names to the set
            for elem in imported_elements:
                module_imports[module_name]['specific_names'].add(elem.name)

        # Check if module is available/installed
        if is_imported_model(module_name):
            models_dir = os.path.join(output_dir, "models")
            filename = os.path.join(models_dir, module_name + ".py")
        else:
            module = import_module_by_name(module_name, output_dir)
            if module is None:
                # Mark this import node so the C++ frontend knows the module was not found
                node.module_not_found = True
                continue

            # Check if module has __file__ attribute (built-in C extensions don't)
            if not hasattr(module, '__file__') or module.__file__ is None:
                # Skip built-in C extension modules (e.g., _sre, _socket, etc.)
                continue

            filename = module.__file__

        # Don't process the file here; we'll do it once after collecting all imports
        node.full_path = filename


def resolve_module_file(module_qualname: str, output_dir: str) -> str | None:
    """Return file path for module qualname (or None if stdlib/missing)."""
    try:
        mod = import_module_by_name(module_qualname, output_dir)
    except SystemExit:
        return None
    filename = mod if isinstance(mod, str) else getattr(mod, "__file__", None)
    if not filename or is_standard_library_file(filename):
        return None
    if not os.path.exists(filename):  # e.g. math.pi is not a submodule
        return None
    return filename


def filter_imports(tree: ast.AST) -> ast.AST:
    """
    Remove import statements for verification-agnostic testing frameworks(import pytest) from the AST.

    This prevents the C++ backend from trying to open JSON files for
    imported testing frameworks that we intentionally skip.
    """
    filtered_body = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            # Filter out frameworks
            filtered_names = []
            for alias in node.names:
                base_module = alias.name.split(".")[0]
                if not is_testing_framework(base_module):
                    filtered_names.append(alias)
            # If all imports were testing frameworks, skip the entire import statement
            if filtered_names:
                node.names = filtered_names
                filtered_body.append(node)

        elif isinstance(node, ast.ImportFrom):
            # Filter out "from testing_framework import ..." statements
            if node.module:
                base_module = node.module.split(".")[0]
                if not is_testing_framework(base_module):
                    filtered_body.append(node)
            else:
                # Relative import without module (from . import x)
                filtered_body.append(node)
        else:
            filtered_body.append(node)

    tree.body = filtered_body
    return tree


def parse_file(filename: str) -> tuple[ast.AST, Preprocessor]:
    """Open, parse, and run Preprocessor on a Python source file.

    Returns the transformed tree alongside the Preprocessor instance so
    callers can read its cross-module export tables (#4525).
    """
    with open(filename, "r", encoding="utf-8") as src:
        tree = ast.parse(src.read())
    preprocessor = Preprocessor(filename)
    tree = preprocessor.visit(tree)
    return tree, preprocessor


def parse_file_canonicalised(filename: str) -> tuple[ast.AST, Preprocessor]:
    """Parse a file and run only the alias-canonicalisation pre-pass.

    Returns the raw tree (with ``apply_range_rewrites`` already applied for
    in-module aliases) plus the Preprocessor instance so the caller can
    later (a) feed it cross-module seeds via ``apply_range_rewrites`` and
    (b) call ``finalize_module`` once propagation has converged. This split
    is the key to #4533: ``visit_For`` must run after every alias known to
    the import graph has been rewritten to ``range``.
    """
    with open(filename, "r", encoding="utf-8") as src:
        tree = ast.parse(src.read())
    preprocessor = Preprocessor(filename)
    preprocessor.prepare_module(tree)
    return tree, preprocessor


def emit_file_as_json(
    filename: str,
    output_dir: str,
    module_qualname: str | None = None,
    elements_to_import=None,
) -> None:
    """Generate AST JSON for a file."""
    tree, _preprocessor = parse_file(filename)
    generate_ast_json(
        tree,
        filename,
        elements_to_import,
        output_dir,
        module_qualname=module_qualname,
    )


def emit_module_json(
    module_qualname: str,
    output_dir: str,
    elements_to_import=None,
) -> None:
    """Resolve module to file and emit AST JSON."""
    filename = resolve_module_file(module_qualname, output_dir)
    if filename:
        emit_file_as_json(filename, output_dir, module_qualname, elements_to_import)


def _snapshot_exports(preprocessor):
    """Snapshot a Preprocessor's range-alias / wrapper export tables (#4525)."""
    return (
        set(preprocessor.exported_range_aliases),
        dict(preprocessor.exported_range_wrappers),
        preprocessor.module_dunder_all,
    )


def _compute_range_seed(module_node):
    """Build a (alias_seed, wrapper_seed) pair for *module_node* (#4525).

    Walks top-level ``ImportFrom`` statements and projects exported aliases
    and wrappers from each source module (looked up in ``module_exports``)
    into the consumer's seed. Honours ``as`` rebinds and ``__all__`` for
    star imports. Bare ``import X`` (qualified attribute access) is out of
    scope.
    """
    alias_seed = set()
    wrapper_seed = {}
    for stmt in module_node.body:
        if not (isinstance(stmt, ast.ImportFrom) and stmt.module):
            continue
        src_exports = module_exports.get(stmt.module)
        if not src_exports:
            continue
        src_aliases, src_wrappers, src_all = src_exports
        if any(a.name == '*' for a in stmt.names):
            visible = (set(src_all) if src_all is not None
                       else {n for n in (set(src_aliases) | set(src_wrappers))
                             if not n.startswith('_')})
            alias_seed |= (set(src_aliases) & visible)
            for w in src_wrappers:
                if w in visible:
                    wrapper_seed[w] = src_wrappers[w]
            continue
        for a in stmt.names:
            bind_name = a.asname or a.name
            if a.name in src_aliases:
                alias_seed.add(bind_name)
            if a.name in src_wrappers:
                wrapper_seed[bind_name] = src_wrappers[a.name]
    return alias_seed, wrapper_seed


def _propagate_range_aliases_across_modules(parsed_trees):
    """Re-apply alias / wrapper rewrites with cross-module seeds (#4525).

    Iterates to a fixed point so chained re-exports (``lib_b`` re-imports
    aliases from ``lib_a``, which its own consumers then inherit) converge.
    """
    while True:
        changed = False
        for module_name, (tree, _filename, preprocessor) in parsed_trees.items():
            alias_seed, wrapper_seed = _compute_range_seed(tree)
            before = _snapshot_exports(preprocessor)
            preprocessor.apply_range_rewrites(tree,
                                              alias_seed=alias_seed,
                                              wrapper_seed=wrapper_seed)
            after = _snapshot_exports(preprocessor)
            if after != before:
                module_exports[module_name] = after
                changed = True
        if not changed:
            return


def process_collected_imports(output_dir):
    """
    Emit AST JSON for every transitively-imported module.

    Discovery and emission are split into two phases so that names added to
    ``module_imports[m]['specific_names']`` by a transitive importer (parsed
    later in the walk) are seen by the emitter for ``m``. A single-phase loop
    would emit ``m``'s JSON the first time it appears, before its full
    specific_names set is known, and later expansions would never be
    re-emitted — causing transitive symbols to silently disappear from the
    JSON the C++ backend reads.
    """
    # Phase 1 — discovery: harvest imports from every reachable module until
    # module_imports stops growing. Each module is parsed and only the
    # alias-canonicalisation pre-pass is run; the full visit (which lowers
    # for-range to a bounded while via visit_For) is deferred to Phase 1c
    # so it sees aliases resolved by cross-module propagation (#4533).
    parsed_trees = {}  # module_name -> (tree, filename, preprocessor)
    visited = set()

    while True:
        pending = set(module_imports.keys()) - visited
        if not pending:
            break
        for module_name in pending:
            visited.add(module_name)
            filename = resolve_module_file(module_name, output_dir)
            if not filename:
                continue
            tree, preprocessor = parse_file_canonicalised(filename)
            parsed_trees[module_name] = (tree, filename, preprocessor)
            module_exports[module_name] = _snapshot_exports(preprocessor)
            for subnode in ast.walk(tree):
                if isinstance(subnode, (ast.Import, ast.ImportFrom)):
                    rewrite_relative_import(subnode, module_name)
                    process_imports(subnode, output_dir)

    # Phase 1b — cross-module range alias / wrapper propagation (#4525).
    _propagate_range_aliases_across_modules(parsed_trees)

    # Phase 1c — finalize each module now that aliases are canonical (#4533).
    for _module_name, (tree, _filename, preprocessor) in parsed_trees.items():
        preprocessor.finalize_module(tree)

    # Phase 2 — emission: module_imports is now stable, so every emitted JSON
    # contains the full set of names any importer ever asked for.
    for module_name, import_info in module_imports.items():
        imported_elements = None if import_info['import_all'] \
            else [ast.alias(name, None) for name in import_info['specific_names']]

        # Submodule guess (e.g. "pkg.sub" referenced as "pkg.sub.name")
        if import_info['specific_names']:
            for name in list(import_info['specific_names']):
                emit_module_json(f"{module_name}.{name}", output_dir)

        if module_name not in parsed_trees:
            continue
        tree, filename, _preprocessor = parsed_trees[module_name]
        generate_ast_json(tree,
                          filename,
                          imported_elements,
                          output_dir,
                          module_qualname=module_name)


def rewrite_relative_import(node, parent_module: str | None):
    """
    Rewrite a relative ImportFrom node to be absolute.

    Example node structure for "from .x import y":
      node.module = "x"
      node.level = 1

    We need to compute the absolute module path based on
    parent_module and node.level, then set node.module to
    the absolute path and reset node.level to 0.
    """
    # node.level indicates the number of leading dots:
    #   from .x import y  → level = 1
    #   from ..x import y → level = 2
    #   from math import y → level = 0 (not relative)
    lvl = getattr(node, "level", 0)
    if lvl <= 0 or not parent_module:
        # Nothing to fix if it's already absolute or we don't know the parent module
        return

    # Split the parent module name into parts, e.g., "pkg.sub" → ["pkg", "sub"]
    parts = parent_module.split(".")

    # Move up "lvl" levels in the module hierarchy.
    # Example:
    #   parent_module = "l.ks", level = 1 → base = "l"
    #   parent_module = "l", level = 1 → base = "l" (fallback if index <= 0)
    idx = len(parts) - lvl
    base = parent_module if idx <= 0 else ".".join(parts[:idx])

    # Rebuild the full absolute module path.
    # Example:  from .ks import foo  →  from l.ks import foo
    node.module = f"{base}.{node.module}" if node.module else base

    # Reset level to 0 since it's no longer a relative import.
    node.level = 0


# pylint: disable-next=too-many-locals,too-many-branches
def generate_ast_json(tree, python_filename, elements_to_import, output_dir, module_qualname=None):
    """
    Generate AST JSON from the given Python AST tree.

    Parameters
    ----------
    tree
        The Python AST tree to serialize.
    python_filename
        The filename of the Python source file the tree was parsed from.
    elements_to_import
        The elements (classes or functions) to be imported from the module,
        or None to include everything.
    output_dir
        The directory to save the generated JSON file in.
    module_qualname
        Fully-qualified module name used to namespace the output filename
        (e.g. ``pkg.sub.mod``); ``None`` means top-level module.

    """
    # Remove verification-agnostic testing framework imports
    tree = filter_imports(tree)

    # Filter elements to be imported from the module
    filtered_nodes = []
    if elements_to_import is not None and elements_to_import:
        # First pass: collect explicitly imported element names
        explicitly_imported = {elem_info.name for elem_info in elements_to_import}

        # Collect all referenced names (functions and classes) from explicitly imported functions/classes
        referenced_names = set()
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                if node.name in explicitly_imported:
                    referenced_names.update(get_referenced_names(node))

        # Second pass: include explicitly imported items and their referenced functions/classes
        for node in tree.body:
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                # Always include ESBMC helper functions
                if node.name in ['ESBMC_range_has_next_', 'ESBMC_range_next_']:
                    filtered_nodes.append(node)
                # Include explicitly imported items
                elif node.name in explicitly_imported:
                    filtered_nodes.append(node)
                # Include functions/classes referenced by imported items
                elif node.name in referenced_names:
                    filtered_nodes.append(node)

            # Include annotated assignments (e.g., x: int = 42)
            elif isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id in explicitly_imported:
                    filtered_nodes.append(node)

            # Preserve Import/ImportFrom nodes: the C++ converter needs them
            # (with the parser-attached ``full_path``/``module_not_found``
            # attributes) to resolve calls into transitively-imported modules.
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                filtered_nodes.append(node)

    # Convert AST to JSON
    ast_json = ast2json_func(
        ast.Module(body=filtered_nodes, type_ignores=[]) if filtered_nodes else tree
    )
    ast_json["filename"] = python_filename
    ast_json["ast_output_dir"] = output_dir
    _tag_bignum_constants(ast_json)

    # Build JSON path
    if module_qualname:
        parts = module_qualname.split(".")
        json_dir = os.path.join(output_dir, *parts[:-1])  # package subdirs
        json_filename = os.path.join(json_dir, f"{parts[-1]}.json")
    else:
        if python_filename.endswith('__init__.py'):
            dir_name = os.path.basename(os.path.dirname(python_filename))
            json_filename = os.path.join(output_dir, f"{dir_name}.json")
        else:
            json_filename = os.path.join(output_dir,
                                         f"{os.path.basename(python_filename[:-3])}.json")

    os.makedirs(os.path.dirname(json_filename), exist_ok=True)

    # Write AST JSON to file
    try:
        with open(json_filename, "w", encoding="utf-8") as json_file:
            json.dump(ast_json, json_file, indent=4, ensure_ascii=False)
    except Exception as e:
        print(f"Error writing JSON file: {e}")


def _emit_submodule_asts(module_dir, base_module, output_dir):
    for root, _dirs, files in os.walk(module_dir):
        for file in files:
            if not file.endswith('.py'):
                continue
            full_path = os.path.join(root, file)
            try:
                with open(full_path, "r", encoding="utf-8") as f:
                    tree = ast.parse(f.read())
            except UnicodeDecodeError:
                continue
            generate_ast_json(tree, full_path, None,
                              f"{output_dir}/{base_module}")


def detect_and_process_submodules(node, processed_submodules, output_dir):
    """
    Detect submodule usage in the AST and process each unseen submodule.

    Parameters
    ----------
    node
        The AST node to scan for submodule attribute accesses.
    processed_submodules
        Set used to avoid reprocessing submodules already handled in this run.
    output_dir
        The directory to save the generated JSON files in.

    """
    if not isinstance(node, ast.Attribute):
        return
    value = node.value
    if not isinstance(value, ast.Name):
        return

    alias = value.id
    base_module = import_aliases.get(alias)

    # Only process submodules of supported model modules
    if not base_module or not is_imported_model(base_module):
        return

    full_module = f"{base_module}.{node.attr}"

    # Avoid reprocessing the same submodule
    if full_module in processed_submodules:
        return
    processed_submodules.add(full_module)

    try:
        module = import_module_by_name(full_module, output_dir)
    except SystemExit:
        return

    file_path = module if isinstance(module, str) else module.__file__
    module_dir = os.path.dirname(file_path)
    _emit_submodule_asts(module_dir, base_module, output_dir)


def check_dependencies():
    """Warn about missing optional dependencies."""
    if shutil.which("mypy") is None:
        print("Warning: mypy not found on PATH; type checking will be skipped.")
        print("  Install with: pip install mypy  or  pipx install mypy")


def select_threading_model(output_dir: str, deadlock_check: bool) -> None:
    """Swap ``models/threading.py`` for the deadlock-aware variant.

    Mirrors the C preprocessor swap of ``pthread_mutex_lock`` to
    ``pthread_mutex_lock_check`` (``clang-c-frontend/c_preprocess.cpp``).
    When ``deadlock_check`` is true, overwrite ``threading.py`` with
    the contents of ``threading_deadlock.py``. The variant file stays
    on disk; the model-JSON loop in ``main`` skips it explicitly by
    name so it does not emit a stray ``Lock``/``Thread`` module.
    """
    if not deadlock_check:
        return
    models_dir = os.path.join(output_dir, "models")
    src = os.path.join(models_dir, "threading_deadlock.py")
    dst = os.path.join(models_dir, "threading.py")
    if os.path.exists(src):
        shutil.copyfile(src, dst)


_RE_ENTRY_FUNCS = frozenset({"match", "search", "fullmatch"})
_RE_MATCH_METHODS = frozenset({"group", "groups", "span"})


def _is_re_match_call(node: ast.AST) -> bool:
    """Return True iff ``node`` is ``re.match/search/fullmatch(pat, str)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    return (isinstance(func.value, ast.Name)
            and func.value.id == "re"
            and func.attr in _RE_ENTRY_FUNCS
            and len(node.args) >= 2)


def _build_re_helper_call(
    helper: str, pat: ast.expr, string: ast.expr, extra_args: list[ast.expr]
) -> ast.Call:
    """Build ``re.<helper>(pat, string, *extra_args)``."""
    return ast.Call(
        func=ast.Attribute(
            value=ast.Name(id="re", ctx=ast.Load()),
            attr=helper,
            ctx=ast.Load(),
        ),
        args=[copy.deepcopy(pat), copy.deepcopy(string), *extra_args],
        keywords=[],
    )


_NESTED_BODY_FIELDS = frozenset({
    "body", "orelse", "handlers", "finalbody", "cases",
})


def _iter_same_scope(node: ast.AST):
    """Yield ``node`` and every descendant in the same lexical scope.

    Stops at nested ``FunctionDef`` / ``AsyncFunctionDef`` / ``Lambda``
    boundaries (variables there are local) and at compound-statement body
    fields (``If.body``, ``For.body``, ...): those nested statement lists
    are processed by :func:`_rewrite_re_calls_in_body` with the bindings
    visible at their entry, so visiting them here would use stale
    outer-scope bindings and silently rewrite an inner-scope ``m.group``
    against the wrong ``(pat, str)`` pair.
    """
    yield node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return
    for field, value in ast.iter_fields(node):
        if field in _NESTED_BODY_FIELDS:
            continue
        if isinstance(value, ast.AST):
            yield from _iter_same_scope(value)
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, ast.AST):
                    yield from _iter_same_scope(item)


def _rewrite_re_calls_in_node(
    node: ast.AST,
    bindings: dict[str, tuple[ast.expr, ast.expr]],
) -> None:
    """In-place rewrite of ``m.group/groups/span(...)`` inside ``node``.

    Walks every same-scope Call descendant; when the receiver name is in
    ``bindings`` and the attribute is one of group/groups/span, replaces
    the Call in place with a direct call to the corresponding re._<method>
    helper. Nested functions/lambdas are not visited here — they are
    enrolled separately by :func:`rewrite_re_match_attribute_calls`.
    """
    for child in _iter_same_scope(node):
        if not isinstance(child, ast.Call):
            continue
        func = child.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in _RE_MATCH_METHODS:
            continue
        if not isinstance(func.value, ast.Name):
            continue
        binding = bindings.get(func.value.id)
        if binding is None:
            continue
        pat, string = binding
        helper = "_" + func.attr
        # Preserve any optional integer argument (group index / span index).
        extra = list(child.args)
        replacement = _build_re_helper_call(helper, pat, string, extra)
        ast.copy_location(replacement, child)
        child.func = replacement.func
        child.args = replacement.args
        child.keywords = replacement.keywords


def _nested_statement_bodies(stmt: ast.stmt) -> list[list[ast.stmt]]:
    """Return the lists of nested statements inside ``stmt`` for recursion."""
    bodies: list[list[ast.stmt]] = []
    if isinstance(stmt, (ast.For, ast.AsyncFor, ast.While, ast.If)):
        bodies.append(stmt.body)
        bodies.append(stmt.orelse)
    elif isinstance(stmt, (ast.With, ast.AsyncWith)):
        bodies.append(stmt.body)
    elif isinstance(stmt, ast.Try):
        bodies.append(stmt.body)
        for handler in stmt.handlers:
            bodies.append(handler.body)
        bodies.append(stmt.orelse)
        bodies.append(stmt.finalbody)
    elif isinstance(stmt, ast.Match):
        for case in stmt.cases:
            bodies.append(case.body)
    return bodies


def _re_call_from_assignment(stmt: ast.stmt) -> ast.Call | None:
    """Return the ``re.<entry>(...)`` Call node bound by ``stmt``, or None.

    Recognises plain assignments (``m = re.match(...)``) and annotated
    assignments with a value (``m: ... = re.search(...)``).
    """
    if isinstance(stmt, ast.AnnAssign):
        value = stmt.value
    elif isinstance(stmt, ast.Assign):
        value = stmt.value
    else:
        return None
    return value if _is_re_match_call(value) else None


def _collect_walrus_re_bindings(
    node: ast.AST,
) -> list[tuple[str, ast.Call]]:
    """Find every ``NamedExpr`` whose value is an ``re.match`` call.

    Returns ``[(target_name, re_call), ...]`` for ``(m := re.match(p, s))``
    occurrences anywhere inside ``node`` (same lexical scope only).
    """
    found: list[tuple[str, ast.Call]] = []
    for child in _iter_same_scope(node):
        if not isinstance(child, ast.NamedExpr):
            continue
        if not isinstance(child.target, ast.Name):
            continue
        if _is_re_match_call(child.value):
            found.append((child.target.id, child.value))
    return found


def _names_from(target: ast.expr) -> list[str]:
    """Return every ``Name`` id bound by ``target`` (including nested tuples)."""
    out: list[str] = []
    _walk_target_names(target, out)
    return out


def _iter_same_function_scope(node: ast.AST):
    """Yield ``node`` and every descendant inside the same function scope.

    Like :func:`_iter_same_scope` but does *not* stop at compound-statement
    body fields, so the visitor sees every assignment in every nested
    branch. Still stops at nested function/lambda boundaries.
    """
    yield node
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.Lambda)):
        return
    for child in ast.iter_child_nodes(node):
        yield from _iter_same_function_scope(child)


def _collect_assigned_names_anywhere(body: list[ast.stmt]) -> set[str]:
    """Return every ``Name`` id bound anywhere within ``body``.

    Used to invalidate parent-scope bindings after recursing into a
    nested branch (if/for/while/with/try/match): a name reassigned
    inside the branch must not retain its pre-branch ``re.match`` cache
    once control re-joins the outer scope.
    """
    names: set[str] = set()
    for stmt in body:
        for sub in _iter_same_function_scope(stmt):
            if isinstance(sub, ast.Assign):
                for tgt in sub.targets:
                    names.update(_names_from(tgt))
            elif isinstance(sub, ast.AnnAssign) and sub.value is not None:
                names.update(_names_from(sub.target))
            elif isinstance(sub, (ast.AugAssign, ast.For, ast.AsyncFor)):
                names.update(_names_from(sub.target))
            elif (isinstance(sub, ast.NamedExpr)
                  and isinstance(sub.target, ast.Name)):
                names.add(sub.target.id)
            elif isinstance(sub, (ast.With, ast.AsyncWith)):
                for item in sub.items:
                    if item.optional_vars is not None:
                        names.update(_names_from(item.optional_vars))
            elif isinstance(sub, ast.ExceptHandler) and sub.name is not None:
                names.add(sub.name)
    return names


def _walk_target_names(target: ast.expr, into: list[str]) -> None:
    """Collect every ``Name`` id bound by ``target`` (handles nested tuples)."""
    if isinstance(target, ast.Name):
        into.append(target.id)
    elif isinstance(target, (ast.Tuple, ast.List)):
        for elt in target.elts:
            _walk_target_names(elt, into)
    elif isinstance(target, ast.Starred):
        _walk_target_names(target.value, into)


def _assigned_names_for_re_binding(stmt: ast.stmt) -> tuple[list[str], list[str]]:
    """Return ``(top_level_names, all_names)`` bound by ``stmt``.

    Only ``top_level_names`` (single-Name targets) can hold the result of
    an ``re.match`` call directly; ``all_names`` are every Name reassigned
    (including tuple-destructure elements) — used to invalidate stale
    bindings on a reassignment that does not point at an ``re.match`` call.
    """
    top: list[str] = []
    every: list[str] = []
    if isinstance(stmt, ast.Assign):
        for tgt in stmt.targets:
            if isinstance(tgt, ast.Name):
                top.append(tgt.id)
            _walk_target_names(tgt, every)
    elif isinstance(stmt, ast.AnnAssign) and stmt.value is not None:
        if isinstance(stmt.target, ast.Name):
            top.append(stmt.target.id)
        _walk_target_names(stmt.target, every)
    return top, every


def _rewrite_re_calls_in_body(
    body: list[ast.stmt],
    parent_bindings: dict[str, tuple[ast.expr, ast.expr]],
) -> None:
    """Rewrite re-match attribute calls inside ``body``.

    Walks statements in order, maintaining a {var: (pat, str)} map of
    variables bound to an ``re.match/search/fullmatch(pat, str)`` call.
    On each statement: first rewrite descendant ``m.group/groups/span``
    calls using the bindings visible *before* the statement executes,
    then update the bindings based on this statement's assignment, then
    recurse into nested bodies with the updated map.
    """
    bindings: dict[str, tuple[ast.expr, ast.expr]] = dict(parent_bindings)
    for stmt in body:
        _rewrite_re_calls_in_node(stmt, bindings)

        # Statement-level assignment to a Name.
        re_call = _re_call_from_assignment(stmt)
        top, every = _assigned_names_for_re_binding(stmt)
        bound: set[str] = set()
        if re_call is not None:
            for name in top:
                bindings[name] = (re_call.args[0], re_call.args[1])
                bound.add(name)
        for name in every:
            if name not in bound:
                bindings.pop(name, None)

        # Walrus assignments (``m := re.match(...)``) anywhere in the stmt.
        for name, call in _collect_walrus_re_bindings(stmt):
            bindings[name] = (call.args[0], call.args[1])

        # Recurse into nested branches with the (now-current) bindings, then
        # invalidate any name reassigned inside any branch so the join below
        # cannot retain a stale (pat, str) pair.
        for nested_body in _nested_statement_bodies(stmt):
            _rewrite_re_calls_in_body(nested_body, bindings)
            for name in _collect_assigned_names_anywhere(nested_body):
                bindings.pop(name, None)


def rewrite_re_match_attribute_calls(tree: ast.Module) -> None:
    """Rewrite ``m.group(N)`` / ``m.groups()`` / ``m.span(N)`` into direct
    calls to ``re._group/_groups/_span(pat, string, ...)`` when ``m`` was
    bound by ``re.match/search/fullmatch(pat, string)``.

    CPython returns a Match-or-None object from these entry points; the
    ESBMC Python frontend cannot soundly model that without breaking
    existing truthiness assertions, so we keep the model boolean and
    redirect the Match accessor methods to side-channel helpers in
    ``models/re.py`` that recompute the result from the original
    ``(pattern, string)`` pair.

    Limitations: the rewriter tracks direct bindings only — aliasing via
    ``m2 = m`` does not propagate the cached ``(pat, string)``, so
    ``m2.group(...)`` is left as an unmodelled attribute call. The
    converter's existing ``Unsupported function`` diagnostic surfaces this.
    """
    _rewrite_re_calls_in_body(tree.body, {})
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            _rewrite_re_calls_in_body(node.body, {})


def main():
    check_usage()
    check_dependencies()
    filename = sys.argv[1]
    output_dir = sys.argv[2]
    deadlock_check = len(sys.argv) == 4 and sys.argv[3] == "--deadlock-check"

    # Type checking input program with mypy.
    returncode, mypy_output = run_mypy_strict(filename)
    if returncode != 0:
        print("\033[93m\nType checking warning:\033[0m")
        print(mypy_output)

    # Add the script directory to the front of the import search path
    script_dir = os.path.dirname(os.path.abspath(filename))
    if script_dir and script_dir not in sys.path:
        sys.path.insert(0, script_dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    select_threading_model(output_dir, deadlock_check)

    # Process and convert AST for main file. The entry script is canonicalised
    # before import discovery and finalised only after cross-module alias /
    # wrapper propagation, so visit_For sees ``range(...)`` everywhere an
    # alias resolves to it (#4533).
    with open(filename, "r", encoding="utf-8") as source:
        tree = ast.parse(source.read())

    preprocessor = Preprocessor(filename)
    preprocessor.prepare_module(tree)
    module_exports["__main__"] = _snapshot_exports(preprocessor)

    # Discover imports first (their nodes are not rewritten by visit_*), so
    # process_collected_imports can build the cross-module export tables that
    # the deferred seed-aware re-rewrite below depends on (#4533).
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            process_imports(node, output_dir)

    process_collected_imports(output_dir)

    # Re-apply range-alias / wrapper rewrites on the entry script using the
    # cross-module export tables built during import processing (#4525), then
    # run the deferred visitor pass so visit_For lowers for-range with the
    # bound it now has access to (#4533).
    alias_seed, wrapper_seed = _compute_range_seed(tree)
    preprocessor.apply_range_rewrites(tree,
                                      alias_seed=alias_seed,
                                      wrapper_seed=wrapper_seed)
    tree = preprocessor.finalize_module(tree)

    # Tag assignments / constants / attribute accesses on the fully-lowered
    # tree so constants introduced by visit_* (e.g. range-loop helpers,
    # dataclass synthesis) receive their esbmc_type_annotation.
    processed_submodules = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            add_type_annotation(node)
        elif isinstance(node, ast.Constant):
            annotate_constant_node(node)
        elif isinstance(node, ast.Attribute):
            detect_and_process_submodules(node, processed_submodules, output_dir)

    reject_unsupported_threading_usage(tree, filename)
    validate_threading_thread_usage(tree, filename)
    lower_threading_thread_usage(tree, filename)

    rewrite_re_match_attribute_calls(tree)

    # Generate JSON from AST for the main file.
    generate_ast_json(tree, filename, None, output_dir)

    _emit_model_jsons(output_dir)


def _emit_model_jsons(output_dir):
    """Generate AST JSON for each memory-model module in ``output_dir/models``."""
    models_dir = os.path.join(output_dir, "models")
    for python_file in glob.glob(os.path.join(models_dir, "*.py")):
        filename = os.path.basename(python_file)
        module_name = filename[:-3]

        # Variant of threading.py; select_threading_model copies it into
        # place under --deadlock-check, so skip it as a separate module
        # to avoid emitting a duplicate Lock/Thread JSON.
        if module_name == "threading_deadlock":
            continue

        if is_imported_model(module_name) and module_name != "typing":
            continue

        with open(python_file, encoding="utf-8") as model:
            model_tree = ast.parse(model.read())
            generate_ast_json(model_tree, filename, None, output_dir)


if __name__ == "__main__":
    main()
