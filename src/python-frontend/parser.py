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
    if len(sys.argv) != 3:
        print("Usage: python astgen.py <file path> <output directory>")
        sys.exit(2)


def is_imported_model(module_name):
    models = [
        "math",
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


# Names from the ``threading`` module that ESBMC currently models. Anything
# else (RLock, Semaphore, Condition, Event, Barrier, Timer, ...) is
# rejected at parse time by ``reject_unsupported_threading_usage`` so we
# never emit a half-modelled concurrency construct that could yield a
# silently wrong verification verdict.
SUPPORTED_THREADING_SYMBOLS = frozenset({"Lock", "Thread"})

# Thread keyword arguments outside MVP scope. The current Thread model
# only supports `target=` and `args=`; the rest are refused at parse time
# with a clear message rather than being silently dropped, which would
# produce a misleading verification verdict.
UNSUPPORTED_THREAD_KWARGS = frozenset({"daemon", "name", "kwargs", "group"})


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
    """Return (module_aliases, thread_aliases) for the ``threading`` module.

    ``module_aliases`` are the names bound by ``import threading [as X]``.
    ``thread_aliases`` are the names bound by
    ``from threading import Thread [as X]`` — these refer to ``Thread``
    directly without a module qualifier.
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
    call_node: ast.Call, module_aliases: set[str], thread_aliases: set[str]
) -> bool:
    """Return True iff ``call_node`` constructs a ``threading.Thread``."""
    func = call_node.func
    if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
        return func.value.id in module_aliases and func.attr == "Thread"
    if isinstance(func, ast.Name):
        return func.id in thread_aliases
    return False


def _expr_is_thread_constructor(
    expr: ast.expr, module_aliases: set[str], thread_aliases: set[str]
) -> bool:
    """Convenience wrapper that narrows ``expr`` to ``ast.Call`` first."""
    if not isinstance(expr, ast.Call):
        return False
    return _is_thread_constructor(expr, module_aliases, thread_aliases)


def _target_name_chain(node: ast.AST) -> str | None:
    """Return dotted form of a ``Name``/``Attribute`` chain or None.

    Examples:
      ``Name('f')``                       → ``"f"``
      ``Attribute(Name('m'), 'f')``       → ``"m.f"``
      ``Attribute(Attribute(Name('a'), 'b'), 'c')`` → ``"a.b.c"``
      Lambdas / calls / arbitrary exprs   → None
    """
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if not isinstance(node, ast.Name):
        return None
    parts.append(node.id)
    return ".".join(reversed(parts))


# pylint: disable-next=too-many-locals,too-many-branches,too-many-statements
def validate_threading_thread_usage(tree: ast.AST, source_filename: str) -> None:
    """Refuse ``threading.Thread`` patterns outside the MVP-supported set.

    The MVP supports a deliberately narrow subset so ESBMC never silently
    weakens the concurrency model:

      * Subclassing ``threading.Thread`` is rejected.
      * Only ``target=`` and ``args=`` keyword arguments are accepted;
        ``daemon=``, ``name=``, ``kwargs=``, ``group=`` are refused.
      * ``target=`` must be a ``Name`` or attribute chain ending in a
        ``Name`` — lambdas, ``Call`` expressions, and runtime-variable
        callables are refused.
      * ``args=`` must be a tuple literal — list / set / runtime variables
        are refused.
      * ``Thread(...)`` may not appear inside a loop body; the desugarer
        synthesises module-level state per construction site, which would
        be overwritten across loop iterations.
    """
    def fail(line: int, message: str) -> None:
        print(f"ERROR: {source_filename}:{line}: {message}")
        sys.exit(4)

    module_aliases, thread_aliases = _collect_thread_aliases(tree)
    if not module_aliases and not thread_aliases:
        return

    def base_is_thread(base: ast.expr) -> bool:
        if isinstance(base, ast.Attribute) and isinstance(base.value, ast.Name):
            return base.value.id in module_aliases and base.attr == "Thread"
        if isinstance(base, ast.Name):
            return base.id in thread_aliases
        return False

    # Reject subclassing: any ClassDef whose base resolves to threading.Thread.
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        for base in node.bases:
            if base_is_thread(base):
                fail(
                    node.lineno,
                    "Subclassing threading.Thread is not supported by ESBMC. "
                    "Use `Thread(target=..., args=...)` directly.",
                )

    # Reject Thread() construction inside loop bodies. Walk every For/While
    # body and flag any Thread call nested in it.
    for parent in ast.walk(tree):
        if not isinstance(parent, (ast.For, ast.AsyncFor, ast.While)):
            continue
        for child in ast.walk(ast.Module(body=parent.body, type_ignores=[])):
            if isinstance(child, ast.Call) and _is_thread_constructor(
                child, module_aliases, thread_aliases
            ):
                fail(
                    child.lineno,
                    "threading.Thread construction inside a loop is not yet "
                    "supported. Construct each Thread at a distinct top-level "
                    "or function-scope site.",
                )

    # Reject Thread() with unsupported kwargs / shapes.
    for call_node in ast.walk(tree):
        if not isinstance(call_node, ast.Call):
            continue
        if not _is_thread_constructor(call_node, module_aliases, thread_aliases):
            continue

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

        # Refuse positional arguments — `Thread(group, target, name, args)` is
        # the real Python signature; ESBMC only accepts the keyword form so
        # the parser doesn't have to track group/name slots.
        if call_node.args:
            fail(
                call_node.lineno,
                "threading.Thread requires `target=` and (optionally) `args=` "
                "as keyword arguments; positional arguments are not supported.",
            )

        if target_value is None:
            fail(
                call_node.lineno,
                "threading.Thread requires `target=<function>`; constructions "
                "without an explicit target are not supported.",
            )
            return  # unreachable; fail() exits — narrows target_value for pyright

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


def _find_thread_var_assigns(
    func_body: list[ast.stmt],
    module_aliases: set[str],
    thread_aliases: set[str],
) -> dict[str, ast.Call]:
    """Return ``{var_name: construction_call}`` for ``var = Thread(...)``.

    Only handles single-target ``Assign`` and ``AnnAssign`` to a plain
    ``Name`` at the function-body top level. Anything more elaborate (tuple
    targets, attribute targets, nested constructions) is left for the
    desugarer to refuse.
    """
    assigns: dict[str, ast.Call] = {}
    for stmt in func_body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(
            stmt.targets[0], ast.Name
        ):
            if _expr_is_thread_constructor(
                stmt.value, module_aliases, thread_aliases
            ):
                # _expr_is_thread_constructor confirmed Call narrowing.
                assert isinstance(stmt.value, ast.Call)
                assigns[stmt.targets[0].id] = stmt.value
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.value is not None
            and _expr_is_thread_constructor(
                stmt.value, module_aliases, thread_aliases
            )
        ):
            assert isinstance(stmt.value, ast.Call)
            assigns[stmt.target.id] = stmt.value
    return assigns


def _count_name_assignments_in_body(body: list[ast.stmt], name: str) -> int:
    """Count direct assignments to ``name`` in ``body`` only.

    Walks only the statements directly in ``body`` — does NOT descend into
    nested ``FunctionDef`` / ``Lambda`` / class bodies. The Thread
    single-def check is per-scope, so an inner function shadowing the
    outer ``t`` must not be counted against the outer site (and vice
    versa).
    """
    count = 0
    for stmt in body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == name:
                    count += 1
        elif isinstance(stmt, ast.AnnAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == name:
                count += 1
        elif isinstance(stmt, ast.AugAssign):
            if isinstance(stmt.target, ast.Name) and stmt.target.id == name:
                count += 1
    return count


class _ThreadDesugarer(ast.NodeTransformer):
    """Rewrite ``threading.Thread`` usage into spawn-thread intrinsics.

    For each ``t = threading.Thread(target=f, args=(a, b, c))`` site
    (index ``N``) the desugarer emits, in place of the construction
    statement::

        __pythread_ended_<N> = 0
        def __pythread_trampoline_<N>() -> None:
            f(<a>, <b>, <c>)              # deep-copied arg expressions
            __ESBMC_atomic_begin()
            __pythread_ended_<N> = 1
            __ESBMC_atomic_end()
            __ESBMC_terminate_thread()
        t = threading.Thread()            # bare instance, holds _tid

    ``t.start()`` becomes
    ``t._tid = __ESBMC_spawn_thread(__pythread_trampoline_<N>)`` and
    ``t.join()`` becomes ``__ESBMC_assume(__pythread_ended_<N> == 1)``.

    The trampoline reads each arg by re-using the construction-site
    expression directly (e.g. ``f(resource)`` literally rather than
    through a per-site copy), so threads sharing an instance all see the
    same memory in the GOTO program — required for race / lock-contention
    exploration. The desugarer therefore refuses Thread variables that
    are reassigned within a scope (the lexical binding from var name to
    trampoline name must be unique).
    """

    def __init__(
        self,
        source_filename: str,
        module_aliases: set[str],
        thread_aliases: set[str],
    ) -> None:
        self._source_filename = source_filename
        self._module_aliases = module_aliases
        self._thread_aliases = thread_aliases
        self._site_counter = 0
        self._current_thread_vars: dict[str, int] = {}

    def _fail(self, line: int | str, message: str) -> None:
        print(f"ERROR: {self._source_filename}:{line}: {message}")
        sys.exit(4)

    def _next_site_id(self) -> int:
        site = self._site_counter
        self._site_counter += 1
        return site

    def _make_trampoline(
        self, site_id: int, target_chain: str, args: list[ast.expr]
    ) -> ast.FunctionDef:
        """Return the synthesised per-site trampoline ``FunctionDef``.

        The body is::

            f(<arg expr 0>, ..., <arg expr n>)
            __ESBMC_atomic_begin()
            __pythread_ended_<N> = 1
            __ESBMC_atomic_end()
            __ESBMC_terminate_thread()

        Each ``<arg expr>`` is a deep copy of the construction-site arg
        expression. The trampoline therefore reads each arg directly from
        the variable the user named at the construction site (e.g.
        ``resource``) instead of through a per-site global copy — so
        multiple threads constructed with the same shared instance all
        operate on the SAME memory in the GOTO program, which is what
        the data-race / lock-contention exploration relies on.

        Caveat: the construction-site variables must therefore still be
        in scope at the trampoline's definition site. The
        ``_make_construction_block`` caller satisfies this by emitting
        the trampoline INTO the construction block — adjacent to the
        original assignment — so the args are reached via the same scope
        chain the user wrote.

        The atomic block + ended flag lets ``join`` wait on the flag with
        ``__ESBMC_assume``. ``__ESBMC_terminate_thread`` is required so
        symex doesn't fall off the end of the trampoline frame (the
        spawn-thread machinery leaves the trampoline's call stack empty).
        """
        target_parts = target_chain.split(".")
        target_expr: ast.expr = ast.Name(id=target_parts[0], ctx=ast.Load())
        for part in target_parts[1:]:
            target_expr = ast.Attribute(value=target_expr, attr=part, ctx=ast.Load())

        # Deep-copy each arg AST so further mutation by NodeTransformer
        # doesn't change the trampoline's view of the construction-site
        # expression.
        arg_exprs: list[ast.expr] = [copy.deepcopy(a) for a in args]
        # Force Load context on every Name inside the copied arg subtree
        # (the original may have been Store-positioned, but the
        # trampoline reads them).
        for arg in arg_exprs:
            for node in ast.walk(arg):
                if isinstance(node, (ast.Name, ast.Attribute, ast.Subscript)):
                    node.ctx = ast.Load()

        trampoline_body: list[ast.stmt] = [
            ast.Expr(
                value=ast.Call(
                    func=target_expr,
                    args=arg_exprs,
                    keywords=[],
                )
            ),
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="__ESBMC_atomic_begin", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
            ),
            ast.Assign(
                targets=[
                    ast.Name(
                        id=f"__pythread_ended_{site_id}", ctx=ast.Store()
                    )
                ],
                value=ast.Constant(value=1),
            ),
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(id="__ESBMC_atomic_end", ctx=ast.Load()),
                    args=[],
                    keywords=[],
                )
            ),
            ast.Expr(
                value=ast.Call(
                    func=ast.Name(
                        id="__ESBMC_terminate_thread", ctx=ast.Load()
                    ),
                    args=[],
                    keywords=[],
                )
            ),
        ]

        trampoline = ast.FunctionDef(
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
            body=trampoline_body,
            decorator_list=[],
            returns=ast.Constant(value=None),
        )
        ast.fix_missing_locations(trampoline)
        return trampoline

    def _make_construction_block(
        self,
        var_name: str,
        site_id: int,
        target_chain: str,
        args: list[ast.expr],
    ) -> list[ast.stmt]:
        """Return the statements that replace ``var = Thread(target=, args=)``.

        Layout:

            __pythread_ended_<N> = 0
            def __pythread_trampoline_<N>(): ...    # reads args by name
            var = Thread()                          # bare instance, _tid=0

        The trampoline references the construction-site arg variables by
        name (rather than copying them through globals), so multiple
        threads constructed with the same shared instance see the same
        memory through the same scope chain — required for race / lock
        exploration to work.
        """
        stmts: list[ast.stmt] = [
            ast.Assign(
                targets=[
                    ast.Name(
                        id=f"__pythread_ended_{site_id}", ctx=ast.Store()
                    )
                ],
                value=ast.Constant(value=0),
            ),
            self._make_trampoline(site_id, target_chain, args),
        ]
        # Bare Thread() construction so the user's variable still holds a
        # Thread instance (with a `_tid` field) the start/join rewrites can
        # read.
        thread_ctor: ast.expr
        if self._module_aliases:
            mod_alias = next(iter(self._module_aliases))
            thread_ctor = ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id=mod_alias, ctx=ast.Load()),
                    attr="Thread",
                    ctx=ast.Load(),
                ),
                args=[],
                keywords=[],
            )
        else:
            thread_alias = next(iter(self._thread_aliases))
            thread_ctor = ast.Call(
                func=ast.Name(id=thread_alias, ctx=ast.Load()),
                args=[],
                keywords=[],
            )
        stmts.append(
            ast.Assign(
                targets=[ast.Name(id=var_name, ctx=ast.Store())],
                value=thread_ctor,
            )
        )
        for s in stmts:
            ast.fix_missing_locations(s)
        return stmts

    def _make_start_call(self, var_name: str, site_id: int) -> ast.AST:
        """Return AST for ``var._tid = __ESBMC_spawn_thread(trampoline_N)``."""
        node = ast.Assign(
            targets=[
                ast.Attribute(
                    value=ast.Name(id=var_name, ctx=ast.Load()),
                    attr="_tid",
                    ctx=ast.Store(),
                )
            ],
            value=ast.Call(
                func=ast.Name(id="__ESBMC_spawn_thread", ctx=ast.Load()),
                args=[
                    ast.Name(
                        id=f"__pythread_trampoline_{site_id}",
                        ctx=ast.Load(),
                    )
                ],
                keywords=[],
            ),
        )
        ast.fix_missing_locations(node)
        return node

    def _make_join_block(self, site_id: int) -> list[ast.AST]:
        """Return the AST that replaces ``t.join()``.

        Emits a single ``__ESBMC_assume(__pythread_ended_<N> == 1)``. The
        assume is not wrapped in an atomic block: blocking context
        switches around it would prevent the scheduler from giving the
        spawned trampoline a chance to flip the flag in the first place.
        """
        node = ast.Expr(
            value=ast.Call(
                func=ast.Name(id="__ESBMC_assume", ctx=ast.Load()),
                args=[
                    ast.Compare(
                        left=ast.Name(
                            id=f"__pythread_ended_{site_id}",
                            ctx=ast.Load(),
                        ),
                        ops=[ast.Eq()],
                        comparators=[ast.Constant(value=1)],
                    )
                ],
                keywords=[],
            )
        )
        ast.fix_missing_locations(node)
        return [node]

    def _rewrite_body(
        self,
        body: list[ast.stmt],
        scope_label: str,
        scope_lineno: int,
    ) -> list[ast.stmt]:
        """Rewrite a function or module body: in-place desugar Thread uses."""
        thread_vars = _find_thread_var_assigns(
            body, self._module_aliases, self._thread_aliases
        )

        # Single-def check inside this scope. Only direct-body assignments
        # count — nested function bodies are independent scopes.
        for var_name in thread_vars:
            if _count_name_assignments_in_body(body, var_name) > 1:
                self._fail(
                    scope_lineno,
                    f"threading.Thread variable `{var_name}` is reassigned in "
                    f"{scope_label}. The Thread model requires a "
                    "single-definition binding so the spawn site can resolve "
                    "the target statically.",
                )

        previous = self._current_thread_vars
        self._current_thread_vars = dict(previous)
        try:
            new_body: list[ast.stmt] = []
            for stmt in body:
                lhs_name = self._thread_assign_target(stmt, thread_vars)
                if lhs_name is not None:
                    call_node = thread_vars[lhs_name]
                    target_value, args_values = self._extract_call_kwargs(call_node)
                    target_chain = _target_name_chain(target_value)
                    assert target_chain is not None  # validated earlier
                    site_id = self._next_site_id()
                    self._current_thread_vars[lhs_name] = site_id
                    new_body.extend(
                        self._make_construction_block(
                            lhs_name, site_id, target_chain, args_values
                        )
                    )
                    continue
                visited = self.visit(stmt)
                # visit_Expr may splice a join() call into multiple stmts.
                if isinstance(visited, list):
                    new_body.extend(visited)
                elif visited is not None:
                    new_body.append(visited)
            return new_body
        finally:
            self._current_thread_vars = previous

    # pylint: disable-next=invalid-name
    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        """Rewrite Thread usage within a function body."""
        node.body = self._rewrite_body(
            node.body, f"function `{node.name}`", node.lineno
        )
        return node

    # pylint: disable-next=invalid-name
    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> ast.AST:
        """Rewrite Thread usage within an async function body."""
        node.body = self._rewrite_body(
            node.body, f"async function `{node.name}`", node.lineno
        )
        return node

    def rewrite_module(self, module: ast.Module) -> None:
        """Rewrite Thread usage at module top level and recurse into defs."""
        module.body = self._rewrite_body(module.body, "module top level", 1)

    @staticmethod
    def _thread_assign_target(
        stmt: ast.AST, thread_vars: dict[str, ast.Call]
    ) -> str | None:
        """Return the LHS name iff ``stmt`` is a Thread construction we own."""
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(
            stmt.targets[0], ast.Name
        ):
            name = stmt.targets[0].id
            if name in thread_vars and stmt.value is thread_vars[name]:
                return name
        elif (
            isinstance(stmt, ast.AnnAssign)
            and isinstance(stmt.target, ast.Name)
            and stmt.value is not None
        ):
            name = stmt.target.id
            if name in thread_vars and stmt.value is thread_vars[name]:
                return name
        return None

    @staticmethod
    def _extract_call_kwargs(
        call_node: ast.Call,
    ) -> tuple[ast.expr, list[ast.expr]]:
        """Pull ``target=`` and ``args=`` from a validated Thread call."""
        target_value: ast.expr | None = None
        args_values: list[ast.expr] = []
        for kw in call_node.keywords:
            if kw.arg == "target":
                target_value = kw.value
            elif kw.arg == "args" and isinstance(kw.value, ast.Tuple):
                args_values = list(kw.value.elts)
        assert target_value is not None  # validated earlier
        return target_value, args_values

    def _thread_method_site(
        self, node: ast.AST
    ) -> tuple[str, int] | None:
        """If ``node`` is ``t.start()``/``t.join()`` on a tracked Thread var,
        return ``(method_name, site_id)``; otherwise ``None``."""
        if not isinstance(node, ast.Call):
            return None
        func = node.func
        if not isinstance(func, ast.Attribute):
            return None
        if not isinstance(func.value, ast.Name):
            return None
        var_name = func.value.id
        if var_name not in self._current_thread_vars:
            return None
        if func.attr not in ("start", "join"):
            return None
        return func.attr, self._current_thread_vars[var_name]

    # pylint: disable-next=invalid-name
    def visit_Expr(self, node: ast.Expr) -> ast.AST | list[ast.AST]:
        """Replace ``t.start()`` / ``t.join()`` statement-level calls.

        ``start`` lowers to a single ``t._tid = __ESBMC_spawn_thread(...)``
        assignment; ``join`` lowers to a 3-statement atomic-assume block,
        which is returned as a list so :class:`ast.NodeTransformer` splices
        it into the parent body in place of the original ``Expr``.
        """
        match = self._thread_method_site(node.value)
        if match is None:
            self.generic_visit(node)
            return node
        method, site_id = match
        assert isinstance(node.value, ast.Call)
        var_name = node.value.func.value.id  # type: ignore[union-attr]
        if method == "start":
            return self._make_start_call(var_name, site_id)
        return self._make_join_block(site_id)



def desugar_threading_thread(tree: ast.Module, source_filename: str) -> None:
    """Apply :class:`_ThreadDesugarer` to ``tree`` in-place."""
    module_aliases, thread_aliases = _collect_thread_aliases(tree)
    if not module_aliases and not thread_aliases:
        return
    desugarer = _ThreadDesugarer(source_filename, module_aliases, thread_aliases)
    desugarer.rewrite_module(tree)
    ast.fix_missing_locations(tree)


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


def main():
    check_usage()
    check_dependencies()
    filename = sys.argv[1]
    output_dir = sys.argv[2]

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
    desugar_threading_thread(tree, filename)

    # Generate JSON from AST for the main file.
    generate_ast_json(tree, filename, None, output_dir)

    # Process and convert AST for memory models
    models_dir = os.path.join(output_dir, "models")

    # Iterate over all .py files in the directory
    for python_file in glob.glob(os.path.join(models_dir, "*.py")):
        filename = os.path.basename(python_file)
        module_name = filename[:-3]

        if is_imported_model(module_name) and module_name != "typing":
            continue

        with open(python_file, encoding="utf-8") as model:
            model_tree = ast.parse(model.read())
            # Generate JSON from AST for the memory models.
            generate_ast_json(model_tree, filename, None, output_dir)


if __name__ == "__main__":
    main()
