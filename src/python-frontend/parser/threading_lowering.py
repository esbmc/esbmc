"""Threading validation and lowering helpers for the Python parser frontend."""

from __future__ import annotations

import ast
import copy
import sys

__all__ = [
    "reject_unsupported_threading_usage",
    "validate_threading_thread_usage",
    "lower_threading_thread_usage",
]

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


def _emit_error(source_filename: str, line: int | str, message: str) -> None:
    """Emit a parser-stage threading error and abort."""
    print(f"ERROR: {source_filename}:{line}: {message}")
    sys.exit(4)


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

    unsupported_message = ("is not yet supported by ESBMC. Only threading.Lock is "
                           "currently modelled; Thread/RLock/Semaphore/Condition/Event/"
                           "Barrier/Timer are tracked as follow-ups to the initial "
                           "threading support.")

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
                    _emit_error(
                        source_filename,
                        node.lineno,
                        "'from threading import *' is not supported; "
                        "import names explicitly so ESBMC can verify "
                        "each one is modelled.",
                    )
                name_aliases[alias.asname or alias.name] = alias.name

    for node in ast.walk(tree):
        offending: str | None = None
        if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                and node.value.id in module_aliases
                and node.attr not in SUPPORTED_THREADING_SYMBOLS):
            offending = node.attr
        elif isinstance(node, ast.Name) and node.id in name_aliases:
            original = name_aliases[node.id]
            if original not in SUPPORTED_THREADING_SYMBOLS:
                offending = original

        if offending is not None:
            _emit_error(
                source_filename,
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
        if any(_base_is_thread(b, module_aliases, thread_aliases) for b in node.bases):
            out[node.name] = node
    return out


def _is_subclass_constructor(
    call_node: ast.Call,
    subclass_names: set[str],
) -> bool:
    """Return ``True`` iff ``call_node`` constructs a tracked Thread subclass."""
    func = call_node.func
    return isinstance(func, ast.Name) and func.id in subclass_names


def _extract_name_binding(stmt: ast.stmt) -> tuple[str | None, ast.expr | None]:
    """Return ``(target_name, value)`` for ``Name = value`` / ``Name: T = value``.

    Returns ``(None, None)`` for tuple-targets, attribute-targets, augmented
    assigns, and any other statement shape. Used wherever we need to ask
    "is this a simple single-name binding, and what does it assign?"
    """
    if (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1
            and isinstance(stmt.targets[0], ast.Name)):
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
    return (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
            and inner.func.id == "super")


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
    return (isinstance(inner, ast.Call) and isinstance(inner.func, ast.Name)
            and inner.func.id == "super" and not inner.args and not inner.keywords)


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
    scope_boundaries = (
        ast.FunctionDef,
        ast.AsyncFunctionDef,
        ast.Lambda,
        ast.ClassDef,
        ast.Module,
    )
    loop_ancestors = (
        ast.For,
        ast.AsyncFor,
        ast.While,
        ast.ListComp,
        ast.SetComp,
        ast.DictComp,
        ast.GeneratorExp,
    )

    cursor = parents.get(id(node))
    while cursor is not None:
        if isinstance(cursor, scope_boundaries):
            return False
        if isinstance(cursor, loop_ancestors):
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
        _emit_error(source_filename, line, message)

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
                        _emit_error(
                            source_filename,
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
    subclass_defs = _collect_thread_subclass_defs(tree, module_aliases, thread_aliases)
    module_top_class_ids = {
        id(node)
        for node in (tree.body if isinstance(tree, ast.Module) else [])
        if isinstance(node, ast.ClassDef)
    }
    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef):
            continue
        if not any(base_is_thread(b) for b in node.bases):
            continue
        if id(node) not in module_top_class_ids:
            _emit_error(
                source_filename,
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
            _emit_error(
                source_filename,
                cls.lineno,
                f"Multiple inheritance is not supported for Thread "
                f"subclass `{cls_name}`. Only "
                f"`class {cls_name}(threading.Thread)` is accepted.",
            )

        # C3: must define run.
        if not any(isinstance(m, ast.FunctionDef) and m.name == "run" for m in cls.body):
            _emit_error(
                source_filename,
                cls.lineno,
                f"Thread subclass `{cls_name}` must define a `run` method.",
            )

        # C4: cannot override start or join. The lowering replaces
        # calls to these; a user override would be silently bypassed.
        for m in cls.body:
            if (isinstance(m, ast.FunctionDef) and m.name in THREAD_OVERRIDE_REJECTED_METHODS):
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
                        _is_super_init_call_stmt(stmt) and stmt.value is inner for stmt in m.body)
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
            is_subclass_ctor = (isinstance(value, ast.Call)
                                and _is_subclass_constructor(value, subclass_names))
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
            if isinstance(value, ast.Call) and _is_thread_constructor(value, module_aliases,
                                                                      thread_aliases):
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
                node, module_aliases, thread_aliases) and _inside_loop_within_scope(node, parents):
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
            if (target_name is not None and isinstance(value, ast.Call)
                    and _is_thread_constructor(value, module_aliases, thread_aliases)
                    and counts.get(target_name, 0) > 1):
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
            if (target_name is not None and target_name not in scope_sites
                    and isinstance(value, ast.Call)
                    and _is_thread_constructor(value, module_aliases, thread_aliases)):
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
        if (target_name is not None and target_name not in out and isinstance(value, ast.Call)
                and _is_subclass_constructor(value, subclass_names)):
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
            raise RuntimeError(f"unexpected Thread() kwarg {kw.arg!r} reached lowering; "
                               "validator gap")
        if kw.arg == "target":
            target_value = kw.value
        elif kw.arg == "args" and isinstance(kw.value, ast.Tuple):
            args_values = list(kw.value.elts)
    if target_value is None:
        # validator guarantees ``target=`` is present
        raise RuntimeError("Thread() construction reached lowering without target= kwarg; "
                           "validator gap")
    return target_value, args_values


def _build_target_call(target_expr: ast.expr, site_id: int, n_args: int) -> ast.expr:
    """Return an AST for ``<target>(<arg0>, ..., <argN-1>)``.

    ``<argI>`` is the module-level global ``__pythread_arg_<site>_<i>``
    populated at construction time. ``<target>`` is a deep copy of the
    original ``target=`` chain so the trampoline calls the same function
    the user named.
    """
    args: list[ast.expr] = [
        ast.Name(id=f"__pythread_arg_{site_id}_{i}", ctx=ast.Load()) for i in range(n_args)
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
        body.append(ast.Global(names=[f"__pythread_arg_{site_id}_{i}" for i in range(n_args)]))
    body.extend([
        ast.Expr(value=_build_target_call(target_expr, site_id, n_args)),
        ast.Expr(value=ast.Call(
            func=ast.Name(id="__pyt_terminate", ctx=ast.Load()),
            args=[],
            keywords=[],
        )),
    ])
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
        ast.Expr(value=ast.Call(
            func=ast.Attribute(
                value=ast.Name(id=var_name, ctx=ast.Load()),
                attr="run",
                ctx=ast.Load(),
            ),
            args=[],
            keywords=[],
        )),
        ast.Expr(value=ast.Call(
            func=ast.Name(id="__pyt_terminate", ctx=ast.Load()),
            args=[],
            keywords=[],
        )),
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
        b for b in class_node.bases if not _base_is_thread(b, module_aliases, thread_aliases)
    ]
    for stmt in class_node.body:
        if not (isinstance(stmt, ast.FunctionDef) and stmt.name == "__init__"):
            continue
        stmt.body = [
            ast.Pass() if _is_super_init_call_stmt(inner) else inner for inner in stmt.body
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
    target = ast.Name(id=f"__pythread_arg_{site_id}_{arg_index}", ctx=ast.Store())

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
        targets=[ast.Name(id=f"__pythread_arg_{site_id}_{arg_index}", ctx=ast.Store())],
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
        return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=args, keywords=[])

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
    return ast.Expr(value=ast.Call(
        func=ast.Name(id="__pyt_join", ctx=ast.Load()),
        args=[ast.Name(id=f"__pythread_tid_{site_id}", ctx=ast.Load())],
        keywords=[],
    ))


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
            ast.Global(names=[f"__pythread_arg_{site_id}_{i}" for i in range(len(args_values))]))
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
            ))
    else:
        # _try_rewrite_statement only dispatches Assign/AnnAssign here.
        raise RuntimeError(f"_rewrite_construction_stmt received unexpected stmt type "
                           f"{type(stmt).__name__}")
    return out


# pylint: disable-next=too-many-locals,too-many-branches
def lower_threading_thread_usage(tree: ast.Module, source_filename: str) -> None:
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
    subclass_defs = _collect_thread_subclass_defs(tree, module_aliases, thread_aliases)
    for cls in subclass_defs.values():
        _strip_thread_inheritance(cls, module_aliases, thread_aliases)

    # Stage 2: collect both site shapes under one continuous site-id
    # space so the generated trampoline / tid names never collide.
    sites_by_scope, next_id = _collect_thread_var_sites(tree, module_aliases, thread_aliases)
    subclass_sites, next_id = _collect_subclass_sites(tree, set(subclass_defs), next_id)

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
                prelude.append(_build_arg_declaration(site_id, i, arg_value))
            prelude.append(_build_tid_declaration(site_id))
            prelude.append(_build_trampoline(site_id, target_expr, len(args_values)))

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
        if isinstance(node, (ast.FunctionDef,
                             ast.AsyncFunctionDef)) and id(node.body) in sites_by_scope
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
        if (isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                and stmt.name in target_names):
            latest_target_def_idx = idx
        if (earliest_user_with_thread is None
                and isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef))
                and id(stmt.body) in inner_thread_scopes):
            earliest_user_with_thread = idx
    if earliest_user_with_thread is not None:
        if latest_target_def_idx >= earliest_user_with_thread:
            offender = tree.body[earliest_user_with_thread]
            _emit_error(
                source_filename,
                offender.lineno,
                "threading.Thread target=<name> must be defined before "
                "the function that constructs the Thread; define the "
                "target above its caller, or move the construction to "
                "module scope.",
            )
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
            raise RuntimeError(f"subclass binding `{var_name}` reached lowering without "
                               f"a module-top binding statement; validator gap")
        subclass_insertions.append((
            binding_idx,
            [
                _build_tid_declaration(site_id),
                _build_subclass_trampoline(site_id, var_name),
            ],
        ))
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
            _rewrite_body_in_place(body, scope_sites, module_aliases, thread_aliases)

    ast.fix_missing_locations(tree)


def _find_binding_stmt_index(tree: ast.Module, call_node: ast.Call) -> int | None:
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
        rewritten = _try_rewrite_statement(stmt, scope_sites, module_aliases, thread_aliases)
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
    if (target_name is not None and target_name in scope_sites
            and _is_thread_construction_with_target_kw(value, module_aliases, thread_aliases)):
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
