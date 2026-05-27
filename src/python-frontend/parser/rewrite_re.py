"""Rewrite `re.match`-bound Match-method calls into model helper calls."""

from __future__ import annotations

import ast
import copy

__all__ = ["rewrite_re_match_attribute_calls"]

_RE_ENTRY_FUNCS = frozenset({"match", "search", "fullmatch"})
_RE_MATCH_METHODS = frozenset({"group", "groups", "span"})
_RE_HELPER_BY_METHOD = {
    "group": "_group",
    "groups": "_groups",
    "span": "_span",
}


def _is_re_match_call(node: ast.AST) -> bool:
    """Return True iff ``node`` is ``re.match/search/fullmatch(pat, str)``."""
    if not isinstance(node, ast.Call):
        return False
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    return (isinstance(func.value, ast.Name) and func.value.id == "re"
            and func.attr in _RE_ENTRY_FUNCS and len(node.args) >= 2)


def _build_re_helper_call(helper: str, pat: ast.expr, string: ast.expr,
                          extra_args: list[ast.expr]) -> ast.Call:
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
    "body",
    "orelse",
    "handlers",
    "finalbody",
    "cases",
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
        helper = _RE_HELPER_BY_METHOD[func.attr]
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


def _collect_walrus_re_bindings(node: ast.AST) -> list[tuple[str, ast.Call]]:
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
            elif (isinstance(sub, ast.NamedExpr) and isinstance(sub.target, ast.Name)):
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
